import Control.Concurrent
import Control.Monad
import System.Environment
import System.IO
import GHC.Conc (ThreadId(ThreadId))

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
  args <- getArgs
  let numConsumers = read (args !! 0) :: Int
  let elemsPerConsumer = read (args !! 1) :: Int

  p2c <- newEmptyMVar
  c2p <- newEmptyMVar
  finalResult <- newEmptyMVar

  -- Fork a producer
  forkIO (producer p2c c2p (numConsumers * elemsPerConsumer) numConsumers finalResult)

  -- Fork numConsumers consumers
  replicateM_ numConsumers (forkIO (consumer p2c c2p elemsPerConsumer))

  -- Join the producer
  result <- takeMVar finalResult
  -- Print the final result
  print result
