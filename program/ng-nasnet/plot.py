import numpy as np
import matplotlib.pyplot as plt
import os.path

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path',type=str, help="log file path")
    args=parser.parse_args()

    val_acc=[]
    train_acc=[]
    with open(args.log_path,'r',encoding='utf-8') as f:
        for line in f:
            if 'val acc:' in line:
                val_acc.append(float(line.split(': ')[1]))
            if 'train acc: ' in line:
                train_acc.append(float(line.split(': ')[1]))

    directory=os.path.dirname(args.log_path)
    filename=os.path.splitext(os.path.basename(args.log_path))[0]

    val_acc=np.array(val_acc)

    print('max acc at epoch {}: {}'.format(np.argmax(val_acc),np.max(val_acc)))

    
    x=np.linspace(0,len(val_acc)-1,len(val_acc))

    
    plt.figure()
    ax=plt.gca()

    ax.yaxis.set_major_locator(plt.MultipleLocator(10))

    # ax.set_ylim(0,100)
    plt.ylim(0,100)
    plt.plot(x,val_acc,label='val acc')
    plt.plot(x,train_acc,label='train acc')
    plt.legend()
    plt.savefig(os.path.join(directory,'{}.png'.format(filename)))
