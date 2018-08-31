




file = open("yolov3.cfg","r")
layers = []
layer = {}


for line in file:
    if line == '\n':
        continue

    if line[0] == '#':
        continue

    line = line.strip('\n')

    if line.startswith('['):
        if len(layer) != 0:
            layers.append(layer)
            layer = {}
        line = line.strip('[]')
        layer["type"] = line
    else:
        key, value  = line.rsplit('=',1)
        
        layer[key] = value
layers.append(layer)

        

print(layers[0])

for val in layers[0]:
    print(val,':',layers[0][val])
    

print(len(layers))
file.close()
