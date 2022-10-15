import tkinter as tk
from tkinter.constants import CENTER
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import messagebox
import random
import os

#initialize
dataset_path = "None"
resFig = np.linspace(-10,10,100)
fig = plt.figure() #定義一個圖像窗口
plt.plot(0, 0, '.') #定義x,y和圖的樣式
plt.title("Full Data Set")
fig.savefig('0.png', dpi = 80)
plt.title("Training Set")
fig.savefig('1.png', dpi = 80)
plt.title("Testing Set")
fig.savefig('2.png', dpi = 80)
plt.close(fig)
trAc = 0.00
teAc = 0.00

#window setting
window = tk.Tk()
window.title('Hw1-108602532-王鼎元')
window.geometry('1140x800')
window.resizable(False, False)

def xyz(data):
    x = []
    y = []
    z = []
    for i in range(len(data)):
        x.append(float(data[i][0]))
        y.append(float(data[i][1]))
        z.append(float(data[i][2]))
    return x, y, z

def draw(x, y, z, name, color1, color2, line, w, title):
    fig = plt.figure() #定義一個圖像窗口
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(len(x)):
        if z[i] == 2:
            x1.append(x[i])
            y1.append(y[i])
        else:
            x2.append(x[i])
            y2.append(y[i])
    
    plt.plot(x1, y1, '.',color = color1)
    plt.plot(x2, y2, '.',color = color2)
    x = np.linspace(min(x)-0.1, max(x)+0.1, 100000)
    if line == 1:
        plt.plot(x, (-w[0]*x-w[2])/w[1], '.', color = "black")
    plt.xlim(min(x)-0.1,max(x)+0.1)
    plt.ylim(min(y)-0.1,max(y)+0.1)
    plt.title(title)
    fig.savefig(name, dpi = 80)
    plt.close(fig)

def training(trainingData, learningRate, divergeCondition):
    w = [random.randint(-1000, 1000)/1000, random.randint(-1000, 1000)/1000, random.randint(-1000, 1000)/1000]
    count = 0
    while True:
        #訓練一次
        for i in range(len(trainingData)):
            tmp = w[0]*trainingData[i][0] + w[1]*trainingData[i][1] + w[2]
            #答案錯誤
            if tmp * (trainingData[i][2]*2-3) < 0:
                #更新權重
                if tmp < 0:
                    w[0] = w[0] + trainingData[i][0]*learningRate
                    w[1] = w[1] + trainingData[i][1]*learningRate
                    w[2] = w[2] + learningRate
                else:
                    w[0] = w[0] - trainingData[i][0]*learningRate
                    w[1] = w[1] - trainingData[i][1]*learningRate
                    w[2] = w[2] - learningRate
        #測試訓練集
        acRate = 0
        z = []
        for i in range(len(trainingData)):
            tmp = w[0]*trainingData[i][0] + w[1]*trainingData[i][1] + w[2]
            if tmp * (trainingData[i][2]*2-3) > 0:
                acRate = acRate + 1
                z.append(-trainingData[i][2]+3)
            else:
                z.append(trainingData[i][2])
        finalAc = 100*acRate/len(trainingData)
        #達到收斂條件
        count = count+1
        if finalAc > divergeCondition or count > 100:
            for i in range(len(trainingData)):
                trainingData[i][2] = z[i]
            return trainingData, w, round(finalAc,2)    

def testing(testingData, w):
    #測試
    acRate = 0
    for i in range(len(testingData)):
        tmp = w[0]*testingData[i][0] + w[1]*testingData[i][1] + w[2]
        if tmp * (testingData[i][2]*2-3) > 0:
            acRate = acRate + 1
            testingData[i][2] = (-testingData[i][2]+3)
    finalAc = round(100*acRate/len(testingData),2)
    return testingData, finalAc

def train(dataset_path, learningRate, divergeCondition):
    global trAc
    global teAc
    #讀檔 & list化
    f = open(dataset_path, 'r')
    data = f.read().split("\n")
    f.close()
    if len(data[-1]) == 0:
        del data[-1]
    for i in range(len(data)):
        data[i] = data[i].split(" ")
        for j in range(2):
            data[i][j] = float(data[i][j])
        data[i][2] = int(data[i][2])
       
    #畫第一張圖
    x, y, z = xyz(data)
    draw(x, y, z, "0.png", 'b', 'c', 0, [0,0], "Full Data Set")
    img = tk.PhotoImage(file='0.png')
    result.configure(image=img)
    os.remove('0.png')
    result.image = img
    # 隨機排序
    random.shuffle(data)
    # 丟進測試 & 訓練資料
    trainingData = data[:2*len(data)//3]
    testingData = data[2*len(data)//3:]
    #訓練
    trainingData, w, trAc = training(trainingData, learningRate, divergeCondition)    
    #繪製訓練結果
    x, y, z, = xyz(trainingData)
    draw(x, y, z, "1.png", 'r', "orange", 1, w, "Training Set")
    #測試
    testingData, teAc = testing(testingData, w)
    #繪製測試結果
    x, y, z = xyz(testingData)
    draw(x, y, z, "2.png", 'limegreen', 'green', 1, w, "Testing Set")
    #、訓練結果圖
    img1 = tk.PhotoImage(file='1.png')
    result1.configure(image=img1)
    os.remove('1.png')
    result1.image = img1
    #、測試結果圖
    img2 = tk.PhotoImage(file='2.png')
    result2.configure(image=img2)
    os.remove('2.png')
    result2.image = img2
    #、正確率label更新
    for i in range(len(w)):
        w[i] = round(w[i], 2)
    trainingAccurancy.configure(text="Training accuracy: "+str(trAc)+"%")
    testingAccurancy.configure(text="Testing accuracy:   "+str(teAc)+"%")
    correctFormula.configure(text="Weight:   ("+str(w[0])+")*x + ("+str(w[1])+")*y + ("+str(w[2])+")")
    

def run():
    global dataset_path,learningRate,divergeCondition
    #collect parameters
    if len(getLearningRate.get()) == 0:
        tk.messagebox.showwarning("Warning", "fill Learning Rate")
    else:
        learningRate = float(getLearningRate.get())
        if len(getDivergeCondition.get()) == 0:
            tk.messagebox.showwarning("Warning", "fill Epoch limit")
        else:
            divergeCondition = float(getDivergeCondition.get())

    #training & plot
    if dataset_path == 'None':
        tk.messagebox.showwarning("Warning", "choose training dataset")
    else:
        train(dataset_path, learningRate, divergeCondition)

def importDataset():
    #open choose file window
    global dataset_path
    dataset_path = filedialog.askopenfilename()
    dataName = dataset_path.split('/')[-1]
    currentDataset.configure(text = dataName)
    currentDataset.place(x=300,y=585)

#run
start = tk.Button(text="Run", command=run)
start.place(x=250,y=720)

#learning rate
learningRateLabel = tk.Label(text = "Learning Rate :")
learningRateLabel.place(x=135,y=640)

getLearningRate = tk.Entry()
getLearningRate.place(x=270,y=640)

#diverge condition
getDivergeCondition = tk.Entry()
getDivergeCondition.place(x=270,y=680)

divergeConditionLabel = tk.Label(text = "Accuracy rate(%) :")
divergeConditionLabel.place(x=135,y=680)

#dataset
importData = tk.Button(text="Choose Dataset", command=importDataset)
importData.place(x=165,y=580)
currentDataset = tk.Label(text = dataset_path)
currentDataset.place(x=300,y=585)

#results
trainingAccurancy = tk.Label(text="Training accuracy:   "+str(trAc)+"%")
trainingAccurancy.place(x=80,y=480)
testingAccurancy = tk.Label(text="Testing accuracy:   "+str(teAc)+"%")
testingAccurancy.place(x=80,y=510)

#formula
correctFormula = tk.Label(text="Weight:   0*x + 0*y + 0")
correctFormula.place(x=280,y=495)

#pictures
img = tk.PhotoImage(file='0.png')
os.remove('0.png')
result = tk.Label(image=img)
result.place(x=30,y=0)

img1 = tk.PhotoImage(file='1.png')
os.remove('1.png')
result1 = tk.Label(image=img1)
result1.place(x=570,y=0)

img2 = tk.PhotoImage(file='2.png')
os.remove('2.png')
result2 = tk.Label(image=img2)
result2.place(x=570,y=400)

window.mainloop()




# 2cring
# 2cs
# 2ring
