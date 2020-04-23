import os
workers=["10.128.0.8","10.128.0.16"]
process = os.popen("sudo scp -i /home/ayush/.ssh/id_rsa -o stricthostkeychecking=no /home/ayush/ADS/ann/binaize-threat-model.pt ayush@%s:/home/ayush/ADS/predict_workers" %(workers[0]))
output=process.read()
process = os.popen("sudo scp -i /home/ayush/.ssh/id_rsa -o stricthostkeychecking=no /home/ayush/ADS/ann/binaize-threat-model.pt ayush@%s:/home/ayush/ADS/predict_workers" %(workers[1]))
output=process.read()

