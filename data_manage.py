import os, sys
import shutil

def move_pictures():
    baseDir = 'E:/workspace/newFaceMethod/data/aligned_images_DB/aligned_images_DB'
    starDirs = [baseDir+'/'+d for d in os.listdir(baseDir)]
    copyBase = 'faces'
    cnt = 0
    for sd in starDirs:
        numDirs = [sd+'/'+d for d in os.listdir(sd)]
        for nd in numDirs:
            pics = [nd+'/'+p for p in os.listdir(nd)]
            cDir = copyBase+'/'+str(cnt)
            if not os.path.exists(cDir):
                os.makedirs(cDir)
            for p in pics:
                name = p.split('/')[-1]
                cp = cDir+'/'+name
                shutil.move(p, cp)
                print(cnt, p, cp)
            cnt += 1

if __name__=='__main__':
    move_pictures()