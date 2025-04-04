import neunet as nn
import numpy as np
import json

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

inp = np.array([[int(i) for i in (text_to_bits('да') + '0'*(112-len(str(text_to_bits('да')))))], [int(i) for i in (text_to_bits('давай') + '0'*(112-len(str(text_to_bits('давай')))))], [int(i) for i in (text_to_bits('не') + '0'*(112-len(str(text_to_bits('не')))))], [int(i) for i in (text_to_bits('дааа') + '0'*(112-len(str(text_to_bits('дааа')))))],
                [int(i) for i in (text_to_bits('нет') + '0'*(112-len(str(text_to_bits('нет')))))],[int(i) for i in (text_to_bits('дада') + '0'*(112-len(str(text_to_bits('дада')))))],[int(i) for i in (text_to_bits('ни') + '0'*(112-len(str(text_to_bits('ни')))))], [int(i) for i in (text_to_bits('низачто'))],
                [int(i) for i in (text_to_bits('ага') + '0'*(112-len(str(text_to_bits('ага')))))]])

out = np.array([[1],[1],[0],[1],[0],[1],[0],[0],[1]])

network = nn.Network([112, 15, 1], [0, 1], l=0.85)

network.back_propagation(1000, inp, out)

inp = input('ans').lower().replace(' ', '')
while inp != '0':
    network.run([int(i) for i in (text_to_bits(inp) + '0'*(112-len(str(text_to_bits(inp)))))])
    inp = input('ans').lower().replace(' ', '')

#network.save_weights()