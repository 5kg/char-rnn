require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-size',100,'number of characters to sample')
cmd:option('-c2py',"pinyin",'number of characters to sample')
cmd:option('-pinyin'," ",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end
torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)

local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

protos = checkpoint.protos
local rnn_idx = #protos.softmax.modules - 1
opt.rnn_size = protos.softmax.modules[rnn_idx].weight:size(2)

local c2py = {}
--for c,i in pairs(vocab) do c2py[i] = '' end
for line in io.lines(opt.c2py) do
    c = tonumber(string.match(line, "^%S+"))
    p = string.match(line, "%S+$")
    c2py[c] = p
end

-- initialize the rnn state
local current_state, state_predict_index
local model = checkpoint.opt.model

local num_layers = checkpoint.opt.num_layers
local states = {}
local current_state
local s_prob = {}
local text = {}
for i = 1,opt.size do
    current_state = {}
    for L=1,checkpoint.opt.num_layers do
        -- c and h for all layers
        local h_init = torch.zeros(1, opt.rnn_size)
        if opt.gpuid >= 0 then h_init = h_init:cuda() end
        table.insert(current_state, h_init:clone())
        table.insert(current_state, h_init:clone())
    end
    states[i] = current_state
    s_prob[i] = 0.0
    text[i] = ""
end
state_predict_index = #current_state -- last one is the top h

protos.rnn:evaluate() -- put in eval mode so that dropout works properly

local candidates
local init = false
for c in opt.pinyin:gmatch'.' do
    candidates = {}
    for sidx, current_state in pairs(states) do
--        print(sidx, text[sidx])
	local next_h = current_state[state_predict_index]
	local log_probs = protos.softmax:forward(next_h)

	local prob, idx = log_probs:sort(2, true)
        prob, idx = prob[1], idx[1]
        local count = 0
	for i=1,#ivocab do
          if c2py[idx[i]] == c then
--              print(sidx, text[sidx] .. ivocab[idx[i]], s_prob[sidx] + prob[i])
              table.insert(candidates, {current_state, s_prob[sidx] + prob[i], idx[i], text[sidx] .. ivocab[idx[i]]})
              count = count + 1
              if count >= opt.size then break end
          end
	end
        -- only use the first state when initlizing
        if not init then init = true; break end
    end

    table.sort(candidates, function(a, b) return a[2] > b[2] end)
--    print(candidates)

    for i = 1,opt.size do
        if (candidates[i] ~= nil) then
            local prev_char = torch.Tensor{candidates[i][3]}
	    local embedding = protos.embed:forward(prev_char)
            current_state = candidates[i][1]
            current_state = protos.rnn:forward{embedding, unpack(current_state)}
	    states[i] = {}
            for _, t in ipairs(current_state) do table.insert(states[i], t:clone()) end
            s_prob[i] = candidates[i][2]
            text[i] = candidates[i][4]
        end
    end
end

for i = 1,opt.size do
    print(text[i])
end
