local Entity, parent = torch.class('nn.Entity', 'nn.Module')

function Entity:__init(bsize, outputSize, type)
  parent.__init(self)
  self.output = torch.Tensor(bsize, outputSize)
  self.bsize = bsize
  local tmp0,tmp1,tmp2
  if type == 'rand' then
    tmp1 = torch.rand(1,outputSize)
  else 
    tmp0 = torch.Tensor(outputSize)
    tmp1 = randomkit.normal(tmp0,0,1)
    tmp1 = torch.reshape(tmp1, 1,outputSize)
  end

  self.bias = torch.repeatTensor(tmp1, bsize, 1)--torch.rand(bsize, outputSize)
  self.gradBias = torch.zeros(bsize, outputSize)
end

function Entity:updateOutput(input)
  self.output:copy(self.bias)
  return self.output
end

function Entity:updateGradInput(input, gradOutput)
  self.gradInput = torch.zeros(input:size()):cuda()
  -- self.gradBias:add(1, gradOutput)
  self.gradBias = torch.sum(gradOutput, 1)
  local gradBias_rep = torch.repeatTensor(self.gradBias, self.bsize, 1)
  self.bias:add(-Entity_FACTOR, gradBias_rep)
  return self.gradInput
end
