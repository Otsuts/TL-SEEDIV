import matplotlib.pyplot as plt
 
plt.rcParams["font.sans-serif"]=['SimHei']
plt.rcParams["axes.unicode_minus"]=False
 
def get_precission_data(file_name):
    y_data = []
    with open(file_name,'r') as f:
        for line in f:
            if 'the loss is' in line:
                y_data.append(float(line.split(' ')[8][:-2]))
        
    return y_data

if __name__ == '__main__':
    file_name = 'logs/mldg/0427_LR0.0005_WD0.0001_PL5e-05_PA10.log' 
    y_data = get_precission_data(file_name)
    for i in range(1,16):
        plt.bar(i,y_data[i-1])
    
    for x, y in enumerate(y_data):
        plt.text(x+1, y+2 , str(round(y,1)), ha='center')
    plt.title("Accuracy on each user(MLDG)")
    plt.xlabel("Test user number")
    plt.xticks(range(1,16), range(1,16))
    plt.ylabel("Accuracy")
    
    plt.savefig(f'results/mldg.png')

