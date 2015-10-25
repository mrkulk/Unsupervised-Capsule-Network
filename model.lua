require 'nngraph'
require 'cunn'
require 'rmsprop'
require 'stn'
require 'GradScale'
require 'IntensityMod'
require 'componentMul'
model = {}

function extract_node(model,id_name)
  for i,n in ipairs(model.forwardnodes) do
    if n.data.module then
      if  n.data.module.forwardnodes then
        ret =extract_node(n.data.module, id_name)
        if ret then
          return ret
        end
      end
    end
    if n.data.annotations.name== id_name then
      return n
    end
  end
end

function transfer_data(x)
  return x:cuda()
end

function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (bsize, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end


function get_transformer(x, fg, encoder_out, params, id, mode)
  local inp = encoder_out --nn.Dropout()(encoder_out)
  local outLayer = nn.Linear(params.rnn_size,6)(inp):annotate{name = 'affines_' .. id}
  if true then
    outLayer.data.module.weight:fill(0)
    local Entity = torch.FloatTensor(6):fill(0)
    if mode == 1 then --randomly arrange the entity affine parameters
      Entity[1]= 1+torch.rand(1)[1]*2
      Entity[5]= 1+torch.rand(1)[1]*2
      Entity[3]=torch.rand(1)[1]*2 - 1
      Entity[6]=torch.rand(1)[1]*2 - 1
    else
      Entity[1]= 1
      Entity[5]= 1
    end
    outLayer.data.module.bias:copy(Entity)
  end

  -- there we generate the grids
  local grid = nn.AffineGridGeneratorBHWD(32,32)(nn.View(2,3)(nn.Identity()(outLayer)))
  -- first branch is there to transpose inputs to BHWD, for the bilinear sampler
  local tranet=nn.Transpose({2,3},{3,4})(x)
  local spanet = nn.BilinearSamplerBHWD()({tranet, grid})
  local sp_out = nn.Transpose({3,4},{2,3})(spanet):annotate{name = 'entity_' .. id}

  if mode == 1 then
    -- transforming fg/bg template with same params as above
    local fg_grid = nn.AffineGridGeneratorBHWD(32,32)(nn.View(2,3)(nn.Identity()(outLayer)))
    local fg_tranet=nn.Transpose({2,3},{3,4})(fg)
    local fg_spanet = nn.BilinearSamplerBHWD()({fg_tranet, fg_grid})
    local fg_template = nn.Transpose({3,4},{2,3})(fg_spanet):annotate{name = 'entity_fg_' .. id}
    return sp_out, fg_template
  else
    return sp_out
  end
end

function create_encoder(params)
  local input_image = nn.Identity()()-- nn.JoinTable(2)({x,prev_canvas})
  -- local enc1 = nn.SpatialMaxPooling(2,2)(nn.ReLU()(nn.SpatialConvolution(1, 64, 3, 3)(input_image)))
  -- local enc2 = nn.SpatialMaxPooling(2,2)(nn.ReLU()(nn.SpatialConvolution(64, 64, 3, 3)(enc1)))
  -- local enc = nn.Linear(64*6*6,params.rnn_size)((nn.Reshape(64*6*6)(enc2)))

  -- local imout = get_transformer(input_image, 0, enc, params, 0, 0) -- get_transformer(params, 0, 0)({input_image, enc})

  local enc1_high = nn.SpatialMaxPooling(2,2)(nn.ReLU()(nn.SpatialConvolution(1, 64, 3, 3)(input_image)))
  local enc2_high = nn.SpatialMaxPooling(2,2)(nn.ReLU()(nn.SpatialConvolution(64, 96, 3, 3)(enc1_high)))
  local affines = nn.Linear(96*6*6,params.rnn_size)((nn.Reshape(96*6*6)(enc2_high)))

  return nn.gModule({input_image}, {affines})
end

function create_network(params)
  local prev_s = nn.Identity()() -- LSTM
  local x = nn.Identity()() --input

  local num_entities = params.num_entities
  local template_width = params.template_width
  local image_width = params.image_width
  local bsize = params.bsize

  --- encoder ---
  local enc_params = create_encoder(params)({x})

  if false then -- TODO: implement recurrent version to handle multiple digits
    local rnn_i = {[0] = nn.Identity()(enc_params)}
    local next_s = {}
    local split = {prev_s:split(2 * params.layers)}
    for layer_idx = 1, params.layers do
      local prev_c         = split[2 * layer_idx - 1]
      local prev_h         = split[2 * layer_idx]
      local dropped        = rnn_i[layer_idx - 1]
      local next_c, next_h = lstm(dropped, prev_c, prev_h)
      table.insert(next_s, next_c)
      table.insert(next_s, next_h)
      rnn_i[layer_idx] = next_h
    end
  else
    rnn_i = {[1] = nn.Tanh()(nn.Linear(params.rnn_size, params.rnn_size)(enc_params))}
    next_s = nn.Identity()(prev_s)
  end

  local sts = {}
  local canvas = {}
  for i=1,params.num_entities do
    sts[i] = {}
    local part;
    if params.dataset == "omniglot" then
      -- part = nn.Entity(bsize, template_width*template_width, 'rand')(x)
      part = nn.ReLU()(nn.Entity(bsize, template_width*template_width, 'rand')(x))
    else
      part = nn.ReLU()(nn.Entity(bsize, template_width*template_width, 'rand')(x))
    end
    -- local part = nn.Log()(nn.AddConstant(1)(nn.Exp()(nn.Entity(bsize, template_width*template_width, 'rand')(x))))

    local part_fg = nn.Sigmoid()(nn.Entity(bsize, template_width*template_width,'rand')(x))--nn.Sigmoid()(nn.Linear(params.rnn_size, template_width*template_width)(rnn_i[params.layers]))
    -- local part_fg = nn.Sigmoid()(nn.Linear(params.rnn_size, template_width*template_width)(rnn_i[params.layers]))
    local part_fg_mem = nn.SpatialUpSamplingNearest(3)(nn.Reshape(1,template_width,template_width)(part_fg))
    local mem = nn.Reshape(1,template_width,template_width)(part)--nn.SpatialUpSamplingNearest(3)(nn.Reshape(1,template_width,template_width)(part))
    local mem_out = mem--nn.Sigmoid()(mem)

    --intensity for each mem 
    local intensity = nn.MulConstant(0.1)(nn.Log()(nn.AddConstant(1)(nn.Exp()(nn.MulConstant(10)(nn.Linear(params.rnn_size, 1)(rnn_i[params.layers])))))):annotate{name = 'intensity_' .. i}  --nn.Sigmoid()(nn.Linear(params.rnn_size, 1)(rnn_i[params.layers]))
    local mem_intensity = nn.IntensityMod()({intensity, mem_out})

    sts[i]["enc_out"] = nn.Identity()(rnn_i[params.layers])
    
    local fg_template
    sts[i]["transformer"], fg_template = get_transformer(mem_intensity, part_fg_mem, sts[i]["enc_out"], params, i, 1) 

    local entity_contribution = sts[i]["transformer"]
    table.insert(canvas, entity_contribution)
  end

  local canvas_out = nn.Reshape(1,image_width,image_width)(nn.MulConstant(1)(nn.Sum(2)(nn.JoinTable(2)(canvas))))
  local err = nn.MSECriterion()({canvas_out, x})

  return nn.gModule({x,prev_s}, {err, nn.Identity()(next_s), canvas_out})
end



function setup(preload, path)
  print("Creating a RNN LSTM network.")
  local core_network
  if preload then
    core_network = torch.load(path .. '/network.t7')
  else
    core_network = create_network(params)
    core_network:cuda()
  end
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.bsize, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.bsize, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.bsize, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))

end



function reset_state()
  for j = 0, params.seq_length do
    for d = 1, 2 * params.layers do
      model.s[j][d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(data)
  g_replace_table(model.s[0], model.start_s)
  reset_state()
  local next_canvas
  for i = 1, params.seq_length do
    local x = data:clone()
    local s = model.s[i - 1]
    model.err[i], model.s[i], new_canvas = unpack(model.rnns[i]:forward({x, s}))
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean(), new_canvas
end

function bp(data)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    local x = data:clone()
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local dnewx = transfer_data(torch.zeros(params.bsize, 32, 32))
    local tmp = model.rnns[i]:backward({x, s},
                                       {derr, model.ds, dnewx})
    g_replace_table(model.ds, tmp[2])

    cutorch.synchronize()
  end

  -- model.norm_dw = paramdx:norm()
  -- if model.norm_dw > params.max_grad_norm then
  --   local shrink_factor = params.max_grad_norm / model.norm_dw
  --   paramdx:mul(shrink_factor)
  -- end

  paramx = rmsprop(paramdx, paramx, config, state)

  -- paramx:add(paramdx:mul(-params.lr))
end

