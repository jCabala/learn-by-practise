import Control.Concurrent
import Control.Monad
import System.IO


threadBody :: String -> MVar () -> IO ()
threadBody name handle = do
  print name
  putMVar handle ()


main = do
  hSetBuffering stdout NoBuffering
  --Two threads, each says their name, then we join them
  handle1 <- newEmptyMVar :: IO (MVar ())
  handle2 <- newEmptyMVar :: IO (MVar ())
  forkIO (threadBody "James" handle1)
  forkIO (threadBody "Pritam" handle2)
  takeMVar handle1
  takeMVar handle2
