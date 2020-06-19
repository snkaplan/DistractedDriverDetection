import pymongo
class DBContext:
    def getCollections(self):
        # mycol = mydb["Fails"]
        # result = mycol.find()
        print(self.mydb.collection_names())
        return self.mydb.collection_names()

    def getDataFromCollection(self,collectionName):
        searchedCol = self.mydb[collectionName]
        result = searchedCol.find()
        return result
        # for x in result:
        #     print(x)
    def insertRow(self,row,colName):
        self.mydb[colName].insert_one(row)


    __instance = None
    @staticmethod 
    def getInstance():
        if DBContext.__instance == None:
            DBContext()
        return DBContext.__instance
    def __init__(self):
        if DBContext.__instance != None:
            # raise Exception("This class is a singleton!")
            print("Singleton Class")
        else:
            DBContext.__instance = self
            self.client = pymongo.MongoClient("mongodb+srv://admin2:LEMDeGL1y0R7Zkt5@cluster0-74qdq.mongodb.net/DistractedDriverDetection?retryWrites=true&w=majority")
            self.mydb = self.client["DistractedDriverDetection"]
        
