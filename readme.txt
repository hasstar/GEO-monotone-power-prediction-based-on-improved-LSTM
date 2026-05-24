Main_trainDL.py是模型的训练、评估的文件，里面没有导入其他的py文件，只导入了库函数。
Main_PlotFig.py是根据训练结果进行画图。
Main_Active_atmospheric.py是根据训练的结果进行大气参数分析的，导出了integrated_active_atmospheric_features.csv。


所有的训练过程都在云端https://cloudstudio.net/a/32963372595630080/edit
训练结果是保存了pth文件，在云端，网址是：https://cloudstudio.net/a/32963372595630080/edit
只需要运行Main_trainDL.py即可，数据是splits文件夹。



我把少量的pth文件下载到了本地VAPpth文件夹。