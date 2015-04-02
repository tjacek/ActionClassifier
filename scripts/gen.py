import os, subprocess

path=r"C:\Users\user\Desktop\kwolek\DynamicActionRecognition\Release\DepthMapBinIOSample.exe"
outpuPatht="C:\\Users\\user\\Desktop\\kwolek\\output\\"

def call(r=3,theta=8,beta=4,dim=50):
    fullOutput=outpuPatht + str(r) +"_"+str(theta) +"_" + str(beta) +"_"+ str(dim) +".arff"
    cmd= path +" " + str(r) +" "+str(theta) +" " + str(beta) +" "+ str(dim) +" "+"test.arf"
    #print("\n"+cmd+"\n");
    #rs=os.system(cmd)
    #print(rs)
    args=[" "+str(r)+" ",str(theta)+" ",str(beta)+" ",str(dim)+" ",fullOutput]
    output = subprocess.Popen([path, args], stdout=subprocess.PIPE).communicate()[0]
    print(output)

def experiment():
    r=[3,5]
    angle=[[8,4],[12,4],[12,8]]
    pca=[0,50,100]
    for r_i in r:
        for angle_i in angle:
            for pca_i in pca:
                call(r_i,angle_i[0],angle_i[1],pca_i)
#experiment()
call(3,12,8,100)
