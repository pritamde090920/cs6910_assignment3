import torch
import torch.nn as nn
import random

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Seq2Seq(nn.Module):
    def __init__(self,encoderFramework,decoderFramework):
        super(Seq2Seq,self).__init__()
        self.encoderFramework=encoderFramework
        self.decoderFramework=decoderFramework
    
    def forward(self,source,target,teacher_force_ratio=0.5):
        batch_sz=source.shape[1]
        target_len=target.shape[0]
        target_vocab_size=self.decoderFramework.outputWordSize

        outputs=torch.zeros(target_len,batch_sz,target_vocab_size).to(device=device)

        _,innerLayer,model=self.encoderFramework(source)
        innerLayer=innerLayer.repeat(self.decoderFramework.layersInDecoder,1,1)
        if self.decoderFramework.modelType=="LSTM":
            model=model.repeat(self.decoderFramework.layersInDecoder,1,1)
        x=target[0]
        for t in range(1,target_len):
            output,innerLayer,model=self.decoderFramework(x,innerLayer,model)            
            outputs[t] = output
            best_guess = output.argmax(dim=1)
            x=target[t] if random.random()<teacher_force_ratio else best_guess

        return outputs