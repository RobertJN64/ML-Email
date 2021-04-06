import copy
import random
import json
#from multiprocessing import Process, Queue, Manager

print("Genetic evolver running...")

#hardwired config
validTestModes = ["Absolute",
                  "Closeness",
                  "SimplePosi"]



#internal helper functions

#scans a database and returns the (first) item with the highest value
#format [[item, val], [item, val], [item,val]]
def getHighest(DB):
    high = 0
    best = None
    brow = None
    for row in DB:
        if row[1] > high:
            high = row[1]
            best = row[0]
            brow = row

    if best is None:
        print("ERROR: all items had value 0")
    return best, brow


#returns list of item given input of locations
def getItems(dataset, itemLocs, count = None):
    itemLocs = copy.deepcopy(itemLocs)
    out = []

    if count is None:
        for loc in itemLocs:
            out.append(dataset[loc])

    else:
        if count > len(itemLocs):
            print("ERROR: Set too small")
            return
        for i in range(0, count):
            loc = itemLocs.pop(random.randint(0, len(itemLocs)-1))
            out.append(dataset[loc])
    return out


#loads a dataset and process metadata
def loadDataset(fname, testSize):
    with open(fname) as f:
        dataset = json.load(f)
    inputs = dataset["inputs"]
    outputs = dataset["outputs"]
    data = dataset["data"]

    if testSize > len(data):
        print("ERROR: Test set too large.")
        return
    
    testset = []
    for i in range(0, testSize):
        j = -1
        while j == -1 or j in testset:
            j = random.randint(0, len(data)-1)
        testset.append(j)

    trainset = []
    for i in range(0, len(data)):
        if i not in testset:
            trainset.append(i)

    return data, trainset, testset, {"inputs": inputs, "outputs": outputs}

def processCSV(fnamein, fnameout, outputKeys, deleteKeys, mode="Delete"):
    delete = ' '.join(deleteKeys)
    outputString = ' '.join(outputKeys)

    with open(fnamein) as f:
        rawfile = f.readlines()

    file = []
    for line in rawfile:
        file.append(line.strip("\n"))

    headers = file[0].split(",")

    csvdata = file[1:]

    data = []
    for line in csvdata:
        items = line.split(",")

        if "" not in items or mode != "Delete": #row is not missing data
            newjson = {}
            for i in range(0, len(items)):
                if headers[i] not in delete: #key is not in delete list
                    val = items[i]
                    if val == "" and mode == "FirstRowReplace":
                        val = csvdata[0].split(",")[i] #grab from first row, not great
                    newjson[headers[i]] = float(val)

            data.append(newjson)

    inputs = {}
    for item in data:
        for key in item:
            if key not in inputs:
                inputs[key] = {"min": item[key], "max": item[key]}
            else:
                if item[key] > inputs[key]["max"]:
                    inputs[key]["max"] = item[key]
                elif item[key] < inputs[key]["min"]:
                    inputs[key]["min"] = item[key]

    outputs = {}
    for item in inputs:
        if str(item) in outputString:
            outputs[item] = inputs[item]

    for item in outputs:
        inputs.pop(item)

    outfile = {"data": data, "inputs": inputs, "outputs": outputs}

    with open(fnameout, 'w') as f:
        json.dump(outfile, f, indent=4)

def runTestSet(fnameTest, obj, fnameOut, mode="SimplePosi"):
    output = []
    with open(fnameTest) as f:
        file = json.load(f)

    data = file["data"]
    inputs = obj.expectedInputs
    outputs = obj.expectedOutputs

    header = ["ID"]


    for ExpectedOutput in outputs:
        header.append(str(ExpectedOutput))


    output.append(",".join(header) + "\n")


    for row in data:
        generatedInput = {}
        for item in inputs:
            generatedInput[item] = row[item]

        # wow, only four lines actually deal with the model (these all are generic functions)
        obj.reset()
        obj.receiveInput(generatedInput)  # not scaled
        obj.process()
        generatedOutput = obj.getOutput()  # scaled
        rowdict = {}
        for expectedOutput in outputs:
            rowdict[str(expectedOutput)] =  generatedOutput[expectedOutput]

        if mode == "SimplePosi":
            for item in rowdict:
                if rowdict[item] > 0:
                    rowdict[item] = 1
                else:
                    rowdict[item] = 0

        rowdict["ID"] = row["ID"]

        outrow = []
        for item in header:
            outrow.append(str(rowdict[item]))

        output.append(','.join(outrow) + "\n")

    with open(fnameOut, "w") as f:
        for i in output:
            f.writelines(i)



#gets evo table using 1/2 split method
def getEvo(count):
    count += 1
    evoTable = []
    while count > 0:
        #print(count)
        val = round(count/2)
        count -= val
        evoTable.append(val)
        if count == 1:
            break
    return evoTable

#evolves a database using specified rate
#user can specify a custom evoTable (error if wrong length)
#throws error if DB is not evolvable
def evolve(DB, evoRate, evoTable=None, settings = None):
    if settings is None:
        settings = {}

    print("New Generation")
    #check if DB is evovable
    sample = DB[0][0]
    evolveAttr = getattr(sample, 'evolve')
    if not callable(evolveAttr):
        print("ERROR: DB is missing attribute evolve")

    if evoTable is None:
        evoTable = getEvo(len(DB))

    #check if evo table is valid
    total = 0
    for i in evoTable:
        total += i

    if total != len(DB):
        print("ERROR: Evolve Table expected " + str(len(DB)) + " and recieved " + str(total))

    #create output db
    newDB = []
    goodObjs = []

    #get best scoring objects
    for i in range(0, len(evoTable)):
        item, row = getHighest(DB)
        goodObjs.append(item)
        DB.remove(row)

    #exact copy
    for item in goodObjs:
        newDB.append([item, 0])

    #similar copy
    for i in range(0, len(evoTable)):
        for j in range(0, evoTable[i] - 1):
            newObj = copy.deepcopy(goodObjs[i]).evolve(evoRate, **settings)
            newDB.append([newObj, 0])
        
    return newDB


#tests a single row of data
def Test_Output(recieved, expected, mode):
    if mode == "Absolute":
        if recieved * expected > 0: #check if pos pos or neg neg
            return 1
        else:
            return 0

    elif mode == "Closeness":
        return 1 - (abs(recieved - expected)/2)
    elif mode == "SimplePosi":
        if (recieved > 0 and expected > 0) or (recieved <= 0 and expected <= 0):
            return 1
        else:
            return 0

    else:
        print("ERROR: Invalid testing mode")

#tests a single object
def Test_Obj(obj, data, testMode):
    score = 0

    inputs = obj.expectedInputs
    outputs = obj.expectedOutputs

    for row in data:
        miniscore = 0
        
        generatedInput = {}
        for item in inputs:
            generatedInput[item] = row[item]
            
        #wow, only four lines actually deal with the model (these all are generic functions)
        obj.reset()
        obj.receiveInput(generatedInput) #not scaled
        obj.process()
        generatedOutput = obj.getOutput() #scaled

        for item in outputs:
            val = row[item]
            if testMode != "SimplePosi":
                val = obj.scale(item, row[item])
            miniscore += Test_Output(generatedOutput[item], val, testMode)

        score += miniscore / len(outputs)

    return score


#testing mode with auto datasets
def Test(DB, dataset, trainset, trainsize, testset, renderSettings = None, testMode = "Absolute", multithreading = False):
    traindata = getItems(dataset, trainset, count=trainsize)
    testdata = getItems(dataset, testset)
    return CustomTest(DB, traindata, testdata, renderSettings = renderSettings, testMode = testMode,
                      multithreading=multithreading)



#testing mode with custom datasets
def CustomTest(DB, traindata, testdata, renderSettings = None, testMode = "Absolute", multithreading=False):
    if testMode not in validTestModes:
        print("ERROR: Did not expect test mode: " + testMode + " Try one of these: " + str(validTestModes))

    if not multithreading:
        for i in range(0, len(DB)):
            DB[i][1] = Test_Obj(DB[i][0], traindata, testMode = testMode)
            renderSettings["func"](DB[i][0], score = DB[i][1]*100/len(traindata), **renderSettings["settings"])
    else:
        print("ERROR: Multithreading is not possible at this time!")

    
        
    obj, row = getHighest(DB)
    truescore = Test_Obj(obj, testdata, testMode = testMode)*100/len(testdata)
    renderSettings["func"](obj, score = row[1]*100/len(traindata), **renderSettings["settings"])

    return DB, obj, row[1]*100/len(traindata), truescore

    
                   
                   
    
    
