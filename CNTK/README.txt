#remove old models
rd Models /S /Q

# start myNet simple regression prediction with CNTK
cntk configFile=myNet.cntk 

# for detailed log in file 
cntk configFile=myNet.cntk traceLevel=1 2>log.txt

# simple LSTM net
cntk configFile=myLstm.cntk
