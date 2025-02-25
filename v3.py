# 변경점
# Mjulti-Head Self Attention 추가
# Skip Connection 추가
# Layer Norm 추가
# Dropout 추가
# 기타 등등...

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyper Parameters
batch_size = 64
context_length = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # 새로 추가!
n_layer = 6 # 모델 크기 확장을 위한 새 변수 추가!
n_head = 6
dropout = 0.2


torch.manual_seed(1337)

# wget 
with open('./datas/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# 문자 하나 당 번호 하나를 매깁니다. 
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

# 문자열과 벡터를 오가는 인코딩/디코딩 함수를 만들어줍니다. 
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    '''
    get_batch: 

    split: 어떤 데이터에 대한 batch를 불러올 건지 결정. 
    'train'인 경우 train_data에 대해서, 아닌 경우 val_data에 대해서 가져옴.
    '''
    # 데이터 불러오기
    data = train_data if split == 'train' else val_data 

    # batch size 개수의 랜덤한 offset 생성. 
    ix = torch.randint(len(data) - context_length, (batch_size,)) 

    # 랜덤 생성한 시작지점 ix에 대한 x, y 가져오기
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # pytorch convention에 따라 parameter가 아니므로 buffer에 등록
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # C : head size
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, C)
        out = wei @ v # (T, T) @ (B, T, C) --> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()

        # 위에서 정의된 head 들을 개수만큼 생성
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) 
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        # 병렬로 수행한 다음 이어붙이기
        return out
    
class FeedForward(nn.Module):
    """ Linear Layer 1개 """
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # 레이어 크기를 4배로 키웠다 줄이는 이유는 논문의 point-wise feed-forward를 참고할 것!
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ 트랜스포머 블럭"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # skip connection도 구현 
        # Layer Normalization도 구현
        # 여기서 논문과의 차이점! Layer Norm을 먼저 수행하고 연산 및 residual
        # pre-norm의 개념 찾아보기
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # 토큰 임베딩 테이블 정의
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)

        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        
        # self.sa_head = Head(n_embd)
        # self.sa_heads = MultiHeadAttention(4, n_embd//4) 
        # self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    # target 여부를 optional로 두어야!
    def forward(self, idx, targets=None):

        '''
        idx: (B, T) tensor. x가 들어갈 위치(input)
        targets: (B, T) tensor. y가 들어갈 위치(target)
        logits: (B, T, C) tensor. 여기서 C는 Channel로, 여기서는 vocab_size를 나타냄. 

        즉, C개의 문자가 각 (B, T)의 예측 target에 해당할 각각의 logit을 나타낸 것. 
        '''
        
        B, T = idx.shape
        # 토큰 임베딩 테이블을 lookup하여 토큰 임베딩 진행
        tok_emb = self.token_embedding_table(idx) # (B, T) -> (B, T, n_embd)

        # 각 위치마다 차례대로 정수를 부여하여 위치 임베딩 진행(Bi-gram이라 지금 도움이 되진 않음.)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (0, 1, ... T - 1) 이 (T, C) 짜리 위치 임베딩 텐서가 됨
        
        # 두 임베딩을 더하여 최종 임베딩 구함
        x = tok_emb + pos_emb # (B, T, n_embd) + (T, n_embd) -> (B, T, n_embd)

        #x = self.sa_head(x) # one-head self-attention 거침

        # multi-head attention을 거침
        #x = self.sa_heads(x)

        # feed-forward 한 번 거침
        # x = self.ffwd(x)

        # attention 및 feed-forward를 3번 거침
        x = self.blocks(x)

        # logit 계산
        logits = self.lm_head(x) #(B, T, n_embd) -> (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # 구한 logit과 target을 비교하여 loss 계산

            # 차원을 맞추기 위한 작업
            # loss를 계산하려면 B, C, T의 shape이어야 함. 
            # logit과 target을 모두 B*T로 차원 축소해서 길게 늘린 뒤 비교
            # 자세한 건 pytorch 공식문서 읽어보기!!!! 
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B*T) # -1
            loss = F.cross_entropy(logits, targets) # Negative Log Likelihood 사용 
            # -ln(1/65) ~= 4.17

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        idx: x
        max_new_tokens: 생성 가능한 최대 토큰 개수수
        '''

        for _ in range(max_new_tokens):
            
            # 중간에 여러 layer가 추가됐으므로 idx 배열 size가 바뀜
            # context_length를 넘지 못하도록 아래 코드 추가!
            idx_cond = idx[:, -context_length:]
            
            # forward로 logit과 loss 계산
            logits, loss = self(idx_cond)

            # 각 batch의 마지막 문자에 대한 예측만 취함(Bi-gram이기 때문!)
            logits = logits[:, -1, :] # (B, T, C) -> (B, C)

            # 여기서 들 수 있는 의문점: 어차피 마지막 문자에 대한 예측만 볼 거면서 왜 
            # 계속 늘어나는 idx를 그대로 forward에 넣고 있을까?
            # 나중에 모델을 N-gram으로 확장할 때 함수의 형태를 최대한 일관되게 하려고 함
            # 연산은 비효율적이지만 공부하는데에는 이렇게 하는게 좋을 듯. 

            # Softmax 거쳐서 확률 계산
            probs = F.softmax(logits, dim = -1) # (B, C)

            # 분포에 따라 추출(예측 생성성)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # 각 batch에 대한 예측 결과를 x의 맨 끝에 append
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)
        
        # 위 loop가 돌면서 idx는 (B, T + 1), (B, T + 2), ... , (B, T + max_new_tokens)의 shape이 됨.
        return idx
    

    
model = BigramLanguageModel()
m = model.to(device)

# optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
context=torch.zeros((1, 1), dtype=torch.long)
print("==================== created prediction ====================")
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
print("============================================================")