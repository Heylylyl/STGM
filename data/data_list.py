import json
import os
def load_file_list_recursion(fpath, result):
    allfilelist = os.listdir(fpath)
    for file in allfilelist:
        filepath = os.path.join(fpath, file)
        if os.path.isdir(filepath):
            load_file_list_recursion(filepath, result)
        else:
            result.append(filepath)
            print(len(result))



def scan(input_path, out_put):
    result_list = []
    load_file_list_recursion(input_path, result_list)
    result_list.sort(reverse=True)

    for i in range(len(result_list)):
        print('{}_{}'.format(i, result_list[i]))

    with open(out_put, 'w') as j:
        json.dump(result_list, j)


projname = "QML"
gtpath = f"D:\hyl\STGM\Antarctic/Antgroundtruth{projname}/"
maskpath = f'D:\hyl\STGM\Antarctic/npymask{projname}/'
result_list = []
for i in range(len(os.listdir(gtpath))):
    file_str = gtpath + str(i)+'.npy'

    result_list.append(file_str)
with open(f'./Antgroundtruth{projname}.txt', 'w') as j:
    json.dump(result_list, j)


result_list = []
for i in range(len(os.listdir(maskpath))):
    file_str = maskpath + str(i)+'.npy'
    result_list.append(file_str)
with open(f'./npymask{projname}.txt', 'w') as j:
    json.dump(result_list, j)



# scan('D:\hyl/fuwuqi\chaofen\chaofen/testlrdata', './testsr.txt')
# scan('D:\hyl/fuwuqi\chaofen\chaofen\masksr', './masksr.txt')

