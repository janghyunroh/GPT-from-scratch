# 변경점
# 1. vocab_size가 전역적으로 정의되어 있으므로 모델에 따로 전달X
# 2. 임베딩 벡터 크기를 별도로 두고 linear layer를 거침. (latent vector 과정을 거침)
# -> logit을 곧바로 계산하지 않음음
# 3. position embedding 추가가

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyper Parameters
batch_size = 32
context_length = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 # 새로 추가!


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

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # 토큰 임베딩 테이블 정의
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
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
            
            # forward로 logit과 loss 계산
            logits, loss = self(idx)

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