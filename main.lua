-- Tejas D Kulkarni
-- Usage: th main.lua

require 'nn'
require 'randomkit'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'cutorch'
require 'xlua'
require 'Base'
require 'optim'
require 'image'
require 'sys'
require 'pl'

--------------------------- Init ------------------------------
params = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -p,--plot                                plot while training
   -r,--lr            (default 0.0005)       learning rate
   -i,--max_epochs    (default 200)           maximum nb of iterations per batch, for LBFGS
   --bsize            (default 100)           bsize
   --image_width      (default 32)           
   --template_width   (default 10)           
   --num_entities     (default 10)           number of entities
   --rnn_size         (default 100)
   --seq_length       (default 1)
   --layers           (default 1)
   --init_weight      (default 0.1)
   --max_grad_norm    (default 5)
]]

Entity_FACTOR = 5e3
require 'Entity'
config = {
    learningRate = params.lr,
    momentumDecay = 0.1,
    updateDecay = 0.01
}
require 'model'

trainLogger = optim.Logger(paths.concat(params.save .. '/', 'train.log'))
testLogger = optim.Logger(paths.concat(params.save .. '/', 'test.log'))

trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData.data = trainData.data/255
testData = mnist.loadTestSet(nbTestingPatches, geometry)
fulldata = torch.zeros(trainData.data:size(1) + testData.data:size(1), 1, 32,32)
fulldata[{{1,trainData.data:size(1)},{},{},{}}] = trainData.data:clone()
fulldata[{{trainData.data:size(1)+1,testData.data:size(1)+trainData.data:size(1) },{},{},{}}] = testData.data:clone()

--------------------------- Helper functions ------------------------------

function get_batch(t, data)
  local inputs = torch.Tensor(params.bsize,1,32,32)
  local k = 1
  for i = t,math.min(t+params.bsize-1,data:size(1)) do
     -- load new sample
     local sample = data[i]
     local input = sample[1]:clone()
     -- local _,target = sample[2]:clone():max(1)
     inputs[{k,1,{},{}}] = input
     k = k + 1
  end
  inputs = inputs:cuda()
  return inputs
end

function init()
  print("Network parameters:")
  print(params)
  reset_state(state)
  local epoch = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  print(fulldata:size())
end

function train()
  for epc = 1,params.max_epochs do
    print('epoch #', epc)
    local cntr = 0
    torch.save(params.save .. '/network.t7', model.rnns[1])
    torch.save(params.save .. '/params.t7', params)
    for t = 1,fulldata:size(1),params.bsize do
      xlua.progress(t, fulldata:size(1))
      -- create mini batch
      local inputs = get_batch(t, fulldata)
      local perp, output = fp(inputs)
      bp(inputs)
      cutorch.synchronize()
      collectgarbage()  
      
      if params.plot and math.fmod(cntr, 20) == 0  then 
        test()
      end

      cntr = cntr + 1
      trainLogger:add{['% perp (train set)'] =  perp}
      trainLogger:style{['% perp (train set)'] = '-'}
      -- trainLogger:plot()
    end
  end
end

function test()
  local test_err = 0
  for tt = 1,1 do--trainData:size(),params.bsize do
    local inputs = get_batch(tt, testData)
    local test_perp, test_output = fp(inputs)
    test_err = test_perp + test_err
    local entity_imgs = {}; entity_fg_imgs={};
    for pp = 1,params.num_entities do
      entity_imgs[pp] = extract_node(model.rnns[1], 'entity_' .. pp).data.module.output:double()
      -- entity_fg_imgs[pp] = extract_node(model.rnns[1], 'entity_fg_' .. pp).data.module.output:double()
    end
    local en_imgs = {}; en_fg_imgs={};
    counter=1
    for bb = 1,MAX_IMAGES_TO_DISPLAY do
      for pp=1,params.num_entities do
        en_imgs[counter] =entity_imgs[pp][bb]
        -- en_fg_imgs[counter] = entity_fg_imgs[pp][bb]
        counter = counter + 1 
      end
    end
    if params.plot then
      window1=image.display({image=test_output[{{1,MAX_IMAGES_TO_DISPLAY},{},{},{}}], nrow=1, legend='Predictions', win=window1})
      window2=image.display({image=inputs[{{1,MAX_IMAGES_TO_DISPLAY},{},{},{}}], nrow=1, legend='Targets', win=window2})
      window3=image.display({image=en_imgs, nrow=params.num_entities, legend='Entities', win=window3})
    end
  end
  testLogger:add{['% perp (test set)'] =  test_err}
  testLogger:style{['% perp (test set)'] = '-'}
end


----------------------- Run --------------------
MAX_IMAGES_TO_DISPLAY = 20 --number of digits to display
setup(false)
init()
train()
