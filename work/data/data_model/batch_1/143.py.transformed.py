
import hashlib as hasher
import datetime as date
class Block():
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.hashBlock()
    def hashBlock(self):
        sha = hasher.sha256()
        sha.update((str(self.index) +
                    str(self.timestamp) +
                    str(self.data) +
                    str(self.previous_hash)).encode())
        return sha.hexdigest()
def createGenesisBlock():
    return Block(0, date.datetime.now(), "Genesis Block", "0")
def nextBlock(lastBlock):
    id = lastBlock.index + 1
    timeStp = date.datetime.now()
    data = "Block number: " + str(id)
    hash256 = lastBlock.hash
    return Block(id, timeStp, data, hash256)
blockchain = [createGenesisBlock()]
previousBlock = blockchain[0]
noOfBlocks = 20
for i in range(0, noOfBlocks):
    addBlock = nextBlock(previousBlock)
    blockchain.append(addBlock)
    previousBlock = addBlock
    print(f"Hash: {addBlock.hash} \n")
