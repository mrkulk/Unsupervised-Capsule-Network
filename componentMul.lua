local componentMul, parent = torch.class('nn.componentMul', 'nn.Module')

function componentMul:__init(...)
   parent.__init(self)
end
 
function componentMul:updateOutput(input)
   self.output = torch.cmul(input[1], input[2]):cuda()
   return self.output
end

function componentMul:updateGradInput(input, gradOutput)
   self.gradInput = {torch.Tensor(input[1]:size()):zero(), torch.Tensor(input[1]:size()):zero()} 
   self.gradInput[1] = torch.cmul(input[2], gradOutput)
   self.gradInput[2] = torch.cmul(input[1], gradOutput)
   return self.gradInput
end
