-- (Preset) Phrase bias
-- Makes certain sequences of tokens more or less likely to appear than normal.
-- Run this script once, then see the .conf file in the same directory as this
-- script for more information.

kobold = require("bridge")()  -- This line is optional and is only for EmmyLua type annotations
local userscript = {}  ---@class KoboldUserScript


---@class HashDeque
-- Data structure that simulates a deque of 32-bit unsigned integers but without
-- storing the actual integers -- rather, it stores a hash of those integers
-- so that it only takes O(1) worst-case time to compare two hashes.  However,
-- it also supports all 4 deque operations in O(1) amortized worst-case time,
-- except removing an integer requires knowledge of the integer at the front or
-- back of the deque.
local HashDeque = {}
local HashDeque_mt = {}
local HashDeque_pow = {}  ---@type table<integer, integer>
local HashDeque_mmi = math.maxinteger == 0x7fffffff and 1041204193 or 1117984489315730401
HashDeque_pow[0] = 1
setmetatable(HashDeque, HashDeque_mt)

---@param array? table<integer, integer>
---@return HashDeque
function HashDeque.new(array)
    local hd = {}  ---@type HashDeque
    for k, v in pairs(HashDeque) do
        hd[k] = v
    end
    hd._hash = 5381
    hd._size = 0
    if array ~= nil then
        for k, v in ipairs(array) do
            v = math.tointeger(v)
            for i = 0, 24, 8 do
                hd._hash = 33*hd._hash + ((v>>i)&0xff)
                hd._size = hd._size + 1
                HashDeque_pow[hd._size] = 33 * HashDeque_pow[hd._size-1]
            end
        end
    end
    setmetatable(hd, getmetatable(HashDeque))
    return hd
end

---@param x integer
---@return nil
function HashDeque:push_back(x)
    x = math.tointeger(x)
    for i = 0, 24, 8 do
        local p = rawget(self, "_size")  ---@type integer
        if HashDeque_pow[p] == nil then
            HashDeque_pow[p] = 33 * HashDeque_pow[p-1]
        end
        rawset(self, "_hash", 33*rawget(self, "_hash") + ((x>>i)&0xff))
        rawset(self, "_size", p + 1)
    end
end

---@param x integer
---@return nil
function HashDeque:pop_back(x)
    x = math.tointeger(x)
    for i = 24, 0, -8 do
        rawset(self, "_hash", HashDeque_mmi*(rawget(self, "_hash") - ((x>>i)&0xff)))
    end
    rawset(self, "_size", rawget(self, "_size") - 4)
end

---@param x integer
---@return nil
function HashDeque:push_front(x)
    x = math.tointeger(x)
    for i = 24, 0, -8 do
        local p = rawget(self, "_size")  ---@type integer
        if HashDeque_pow[p] == nil then
            HashDeque_pow[p] = 33 * HashDeque_pow[p-1]
        end
        rawset(self, "_hash", rawget(self, "_hash") + HashDeque_pow[p]*(((x>>i)&0xff) + 172192))
        rawset(self, "_size", p + 1)
    end
end

---@param x integer
---@return nil
function HashDeque:pop_front(x)
    x = math.tointeger(x)
    for i = 0, 24, 8 do
        local p = rawget(self, "_size") - 1
        rawset(self, "_hash", rawget(self, "_hash") - HashDeque_pow[p]*(((x>>i)&0xff) + 172192))
        rawset(self, "_size", p)
    end
end

---@param a HashDeque
---@param b HashDeque
---@return boolean
function HashDeque_mt.__eq(a, b)
    return rawget(a, "_hash") == rawget(b, "_hash")
end


---@class PhraseBiasEntry
---@field starting_bias number
---@field ending_bias number
---@field tokens table<integer, integer>
---@field index table<integer, integer>

local example_config = [[# Phrase bias
#
# For each phrase you want to bias, add a new line into
# this config file as a comma-separated list in this format:
# <starting bias>, <ending bias>, <comma-separated list of token IDs>
# For <starting bias> and <ending bias>, this script accepts floating point
# numbers or -inf, where positive bias values make it more likely for tokens
# to appear, negative bias values make it less likely and -inf makes it
# impossible.
#
# Example 1 (makes it impossible for the word "CHAPTER", case-sensitive, to
# appear at the beginning of a line in the output):
# -inf, -inf, 41481
#
# Example 2 (makes it unlikely for the word " CHAPTER", case-sensitive, with
# a leading space, to appear in the output, with the unlikeliness increasing
# even more if the first token " CH" has appeared):
# -10.0, -20.0, 5870, 29485
#
# Example 3 (makes it more likely for " let the voice of love take you higher",
# case-sensitive, with a leading space, to appear in the output, with the
# bias increasing as each consecutive token in that phrase appears):
# 7, 25.4, 1309, 262, 3809, 286, 1842, 1011, 345, 2440
#
]]

-- If config file is empty, write example config
local f = kobold.get_config_file()
if f:read(1) == nil then
    f:write(example_config)
end
f:close()
example_config = nil

-- Read config
print("Loading phrase bias config...")
local bias_array = {}  ---@type table<integer, PhraseBiasEntry>
f = kobold.get_config_file()
local bias_array_count = 0
local val_count = 0
local line_count = 0
local row = {}  ---@type PhraseBiasEntry
local val_orig
for line in f:lines("l") do
    line_count = line_count + 1
    if line:find("^ *#") == nil and line:find("%S") ~= nil then
        bias_array_count = bias_array_count + 1
        val_count = 0
        row = {}
        row.tokens = {}
        for val in line:gmatch("[^,%s]+") do
            val_count = val_count + 1
            val_orig = val
            if val_count <= 2 then
                val = val:lower()
                if val:sub(-3) == "inf" then
                    val = math.tointeger(val:sub(1, -4) .. "1")
                    if val ~= val or type(val) ~= "number" or val > 0 then
                        error("First two values of line " .. line_count .. " of config file must be finite floating-point numbers or -inf, but got '" .. val_orig .. "' as value #" .. val_count)
                    end
                    val = val * math.huge
                else
                    val = tonumber(val)
                    if val ~= val or type(val) ~= "number" then
                        error("First two values of line " .. line_count .. " of config file must be finite floating-point numbers or -inf, but got '" .. val_orig .. "' as value #" .. val_count)
                    end
                end
                if val_count == 1 then
                    row.starting_bias = val
                else
                    row.ending_bias = val
                end
            else
                val = math.tointeger(val)
                if type(val) ~= "number" or val < 0 then
                    error("All values after the first two values of line " .. line_count .. " of config file must be nonnegative integers, but got '" .. val_orig .. "' as value #" .. val_count)
                end
                row.tokens[val_count - 2] = val
            end
        end
        if val_count < 3 then
            error("Line " .. line_count .. " of config file must contain at least 3 values, but found " .. val_count)
        end
        bias_array[bias_array_count] = row
    end
end
f:close()
print("Successfully loaded " .. bias_array_count .. " phrase bias entr" .. (bias_array_count == 1 and "y" or "ies") .. ".")


local genmod_run = false

---@param starting_val number
---@param ending_val number
---@param factor number
---@return number
local function logit_interpolate(starting_val, ending_val, factor)
    -- First use the logistic function on the start and end values
    starting_val = 1/(1 + math.exp(-starting_val))
    ending_val = 1/(1 + math.exp(-ending_val))

    -- Use linear interpolation between these two values
    local val = starting_val + factor*(ending_val - starting_val)

    -- Return logit of this value
    return math.log(val/(1 - val))
end


function userscript.genmod()
    genmod_run = true

    local context_tokens = kobold.encode(kobold.worldinfo:compute_context(kobold.submission))
    local factor  ---@type number
    local next_token  ---@type integer
    local tokens  ---@type table<integer, integer>
    local n_tokens  ---@type integer

    local biased_tokens = {}  ---@type table<integer, table<integer, boolean>>
    for i = 1, kobold.generated_rows do
        biased_tokens[i] = {}
    end

    -- For each phrase bias entry in the config file...
    for _, bias_entry in ipairs(bias_array) do

        -- For each partially-generated sequence...
        for i, generated_row in ipairs(kobold.generated) do

            -- Build an array `tokens` as the concatenation of the context
            -- tokens and the generated tokens of this sequence

            tokens = {}
            n_tokens = 0
            for k, v in ipairs(context_tokens) do
                n_tokens = n_tokens + 1
                tokens[n_tokens] = v
            end
            for k, v in ipairs(generated_row) do
                n_tokens = n_tokens + 1
                tokens[n_tokens] = v
            end

            -- Determine the largest integer `max_overlap` such that the last
            -- `max_overlap` elements of `tokens` equal the first `max_overlap`
            -- elements of `bias_entry.tokens` by using the Rabin-Karp algorithm

            local max_overlap = 0
            local hd_tokens = HashDeque.new()
            local hd_bias_entry_tokens = HashDeque.new()
            local matches = {}  ---@type table<integer, integer>
            local n_matches = 0
            for j = 1, math.min(n_tokens, #bias_entry.tokens) do
                hd_tokens:push_front(tokens[n_tokens - j + 1])
                hd_bias_entry_tokens:push_back(bias_entry.tokens[j])
                if hd_tokens == hd_bias_entry_tokens then
                    n_matches = n_matches + 1
                    matches[n_matches] = j
                end
            end
            for j = n_matches, 1, -1 do
                for k = 1, matches[j] do
                    if tokens[n_tokens - matches[j] + k] ~= bias_entry.tokens[k] then
                        goto rk_continue
                    end
                end
                max_overlap = matches[j]
                break
                ::rk_continue::
            end

            -- Use `max_overlap` to determine which token in the bias entry to
            -- apply bias to

            if max_overlap == 0 or max_overlap == #bias_entry.tokens then
                if bias_entry.tokens[2] == nil then
                    factor = 1
                else
                    factor = 0
                end
                next_token = bias_entry.tokens[1]
            else
                factor = max_overlap/(#bias_entry.tokens - 1)
                next_token = bias_entry.tokens[max_overlap+1]
            end

            -- Apply bias

            if not biased_tokens[i][next_token] then
                kobold.logits[i][next_token + 1] = kobold.logits[i][next_token + 1] + logit_interpolate(bias_entry.starting_bias, bias_entry.ending_bias, factor)
                biased_tokens[i][next_token] = true
            end
        end
    end
end

function userscript.outmod()
    if not genmod_run then
        warn("WARNING:  Generation modifier was not executed, so this script has had no effect")
    end
end

return userscript
