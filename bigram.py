import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyper Parameters






torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # 토큰 임베딩 테이블 정의
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # target 여부를 optional로 두어야!
    def forward(self, idx, targets=None):

        '''
        idx: (B, T) tensor. x가 들어갈 위치(input)
        targets: (B, T) tensor. y가 들어갈 위치(target)
        logits: (B, T, C) tensor. 여기서 C는 Channel로, 여기서는 vocab_size를 나타냄. 

        즉, C개의 문자가 각 (B, T)의 예측 target에 해당할 각각의 logit을 나타낸 것. 
        '''
        
        # 토큰 임베딩 테이블을 lookup하여 logit 구함
        logits = self.token_embedding_table(idx)

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
    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb) # forward 함수 실행됨
print(logits.shape)
print(loss)