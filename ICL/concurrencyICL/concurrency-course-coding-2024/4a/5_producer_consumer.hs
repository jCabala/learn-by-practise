import Control.Concurrent
import Control.Monad
import System.IO

sendToConsumers :: MVar Int -> Int -> IO ()
sendToConsumers p2c numElems = do
  replicateM_ numElems (putMVar p2c 1) :: IO ()

sumFromChannel :: MVar Int -> Int -> IO Int
sumFromChannel channel numElems = do
  results <- replicateM numElems (takeMVar channel) :: IO [Int]
  return (sum results)

producer :: MVar Int -> MVar Int -> Int -> Int -> MVar Int -> IO ()
producer p2c c2p numElemsToProduce numConsumers finalResult = do
  sendToConsumers p2c numElemsToProduce
  result <- sumFromChannel c2p numConsumers
  putMVar finalResult result

consumer :: MVar Int -> MVar Int -> Int -> IO ()
consumer p2c c2p elemsPerConsumer = do
  myResult <- sumFromChannel p2c elemsPerConsumer
  putMVar c2p myResult

main = do

  p2c <- newEmptyMVar :: IO (MVar Int)
  c2p <- newEmptyMVar :: IO (MVar Int)
  finalResult <- newEmptyMVar :: IO (MVar Int)

  -- 1 producer, producing 200 integers, collecting a result from each producer
  -- and summing the results.
  forkIO (producer p2c c2p 20000 2 finalResult)

  -- 2 consumers, each consuming 100 integers and sending their sum back
  forkIO (consumer p2c c2p 10000)
  forkIO (consumer p2c c2p 10000)

  resultToPrint <- takeMVar finalResult

  print resultToPrint
