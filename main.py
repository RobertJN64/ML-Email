#imports
import MachineLearning.GeneticEvolution as ge
import MachineLearning.GeneticNets as gn
import MachineLearning.NetRender as nr
import EmailManager
import ReportGenerator

import smtplib, ssl
from imbox import Imbox
import shutil

from os import path as ospath, makedirs as mkdrs
from time import sleep

#region CONFIG
#config
allowed_files = ["traindata.json", "testdata.csv", "testdata.json", "rawdata.csv", "template.txt", "config.json"]

#connect to alt email account
host = "imap.gmail.com"
username = "robertjnml@gmail.com"
password = 'robert4ml!'
download_folder = "C:/users/rober/PycharmProjects/ML-Email/files"

port = 465  # For SSL
context = ssl.create_default_context()
smtp_server = "smtp.gmail.com"

#config
fname = "titanic.json"
generations = 10
populationSize = 20
defTrainSize = 400
testSize = 100
evoRate = 4

#specific config
midDepth = 10
midWidth = 1
#endregion

if not ospath.isdir(download_folder):
    mkdrs(download_folder, exist_ok=True)

while True:
    mail = Imbox(host, username=username, password=password, ssl=True, ssl_context=None, starttls=False)
    messages = mail.messages(sent_to='robertjnml+auto@gmail.com', unread=True)  # defaults to inbox

    for uid, message in messages:
        #region Extract Files
        mail.mark_seen(uid)
        print(message.subject)
        attachments = message.attachments
        sender = message.sent_from[0]['email']

        names = []

        for attachment in attachments:
            names.append(attachment["filename"])

        if "traindata.json" not in names and "rawdata.csv" not in names:
            print("Message: ", message.subject, " does not contain an attachment: traindata.json or rawdata.json")
            break

        if "testdata.csv" in names or "testdata.json" in names:
            runTestSet = True
        else:
            runTestSet = False

        projectFolder = download_folder + "/" + str(message.subject).replace(" ", "_")
        #Okay, we are good to process this message
        if not ospath.isdir(projectFolder):
            mkdrs(projectFolder, exist_ok=True)
        else:
            print("PANIC! This folder already exists!")
            break

        for attachment in attachments:
            if attachment["filename"] not in allowed_files:
                print("Attachment file name: ", attachment["filename"], " not allowed!")
            else:
                filepath = projectFolder + "/" + attachment["filename"]
                with open(filepath, "wb") as fp:
                    fp.write(attachment.get('content').read())

        if "rawdata.csv" in names:
            ge.processCSV(projectFolder + '/rawdata.csv', projectFolder + '/traindata.json', ["Output"], ["Ignore"], mode="FirstRowReplace")

        #endregion
        #region okay, ready to start machine learning!
        try:
            dataset, trainset, testset, metadata = ge.loadDataset(projectFolder + "/traindata.json", testSize)

            trainSize = min(defTrainSize, len(trainset)-10)

        except Exception as e:
            print("Dataset loading error: ", str(e))
            break

        try:
            DB = gn.Random(metadata.inputs, metadata.outputs, populationSize, midWidth, midDepth, bias=True)
            bests = []

        except Exception as e:
            print("Error starting machine learning: ", str(e))
            break

        try:
            screen = nr.screen()
            rSettings = nr.stdSettings(screen)
            rSettings["settings"]["vdis"] = 15
            for genCount in range(0, generations):
                evoRate = evoRate * 0.95
                DB, best, bestscore, truescore = ge.Test(DB, dataset, trainset, trainSize, testset,
                                                         renderSettings=rSettings, testMode="SimplePosi")
                bests.append([best, truescore])
                DB = ge.evolve(DB, evoRate)
                screen.bestNet(best, bestscore, truescore)
            nr.stop()
        except Exception as e:
            print("Error during learning: ", str(e))
            break

        try:
            best, row = ge.getHighest(bests)
            gn.saveNets([best], projectFolder + "/net-save", "Nets from file: " + fname, 2)

        except Exception as e:
            print("Error during saving file: ", str(e))
            break

        #endregion

        try:
            if runTestSet:
                best = gn.loadNets(projectFolder + "/net-save")[0][0]
                if "testdata.csv" in names:
                    ge.processCSV(projectFolder + '/testdata.csv', projectFolder + '/testdata.json', ["Output"], ["Ignore"], mode="FirstRowReplace")
                ge.runTestSet(projectFolder + "/testdata.json", best, projectFolder + "/submission.csv")

        except Exception as e:
            print("Error during test set: ", str(e))
            break

        #try:
        configfolder = "ReportSettings/config.json"
        if "config.json" in names:
            configfolder = projectFolder + "/config.json"
        templatefoler = "ReportSettings/Template.txt"
        if "template.txt" in names:
            templatefoler = projectFolder + "/template.txt"
        ReportGenerator.GenerateReport(projectFolder, configfolder, templatefoler)

        #except Exception as e:
            #print("Error during report generation: ", str(e))
            #break

        try:
            attachmentsList = [projectFolder + "/net-save.json"]
            if runTestSet:
                attachmentsList.append(projectFolder + "/submission.csv")
            shutil.make_archive(projectFolder + "/ReportZip", 'zip', projectFolder+"/Report")
            attachmentsList.append(projectFolder + "/ReportZip.zip")

            text = "We have finished your machine learning project. Files are attached."
            message_to_send = EmailManager.createEmail("Machine Learning Project: " + message.subject + " finished!", text, username, sender, attachmentsList)

        except Exception as e:
            print("Error creating email: ", str(e))
            break

        try:
            with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                server.login(username, password)
                server.sendmail(username, str(sender), message_to_send)
            print("Email sent!")

        except Exception as e:
            print("Error during sending email: ", str(e))
            break

    mail.logout()
    sleep(5)


