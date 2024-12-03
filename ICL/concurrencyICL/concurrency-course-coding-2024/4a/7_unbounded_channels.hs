import Control.Concurrent
import Control.Monad
import System.Environment
import System.IO

type Stream a = MVar (Item a)

data Item a = Item a (Stream a)

data Channel a = Channel (MVar (Stream a)) (MVar (Stream a))

newChannel :: IO (Channel a)
newChannel = do
  emptyStream <- newEmptyMVar
  readEnd <- newMVar emptyStream
  writeEnd <- newMVar emptyStream
  return (Channel readEnd writeEnd)

readChannel :: Channel a -> IO a
readChannel (Channel readEnd _) = do
  readStream <- takeMVar readEnd
  (Item value remainder) <- takeMVar readStream
  putMVar readEnd remainder
  return value

writeChannel :: Channel a -> a -> IO ()
writeChannel (Channel _ writeEnd) value = do
  newEmptyStream <- newEmptyMVar
  exisingWriteEndStream <- takeMVar writeEnd
  putMVar exisingWriteEndStream (Item value newEmptyStream)
  putMVar writeEnd newEmptyStream











sendToConsumers :: Channel Int -> Int -> IO ()
sendToConsumers p2c numElems = do
  replicateM_ numElems (writeChannel p2c 1) :: IO ()

sumFromChannel :: Channel Int -> Int -> IO Int
sumFromChannel channel numElems = do
  results <- replicateM numElems (readChannel channel) :: IO [Int]
  return (sum results)

producer :: Channel Int -> Channel Int -> Int -> Int -> Channel Int -> IO ()
producer p2c c2p numElemsToProduce numConsumers finalResult = do
  sendToConsumers p2c numElemsToProduce
  result <- sumFromChannel c2p numConsumers
  writeChannel finalResult result

consumer :: Channel Int -> Channel Int -> Int -> IO ()
consumer p2c c2p elemsPerConsumer = do
  myResult <- sumFromChannel p2c elemsPerConsumer
  writeChannel c2p myResult

main = do
  args <- getArgs
  let numConsumers = read (args !! 0) :: Int
  let elemsPerConsumer = read (args !! 1) :: Int

  p2c <- newChannel
  c2p <- newChannel
  finalResult <- newChannel

  -- Fork a producer
  forkIO (producer p2c c2p (numConsumers * elemsPerConsumer) numConsumers finalResult)

  -- Fork numConsumers consumers
  replicateM_ numConsumers (forkIO (consumer p2c c2p elemsPerConsumer))

  -- Join the producer
  result <- readChannel finalResult
  -- Print the final result
  print result
