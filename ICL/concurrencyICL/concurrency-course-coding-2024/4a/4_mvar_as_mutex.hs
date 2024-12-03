import Control.Concurrent
import Control.Monad
import System.IO

protectedPrint :: String -> MVar() -> MVar() -> IO ()
protectedPrint value m1 m2 = do
  putMVar m1 ()
  putMVar m2 ()
  print value
  takeMVar m2
  takeMVar m1

threadBody :: String -> MVar() -> MVar () -> MVar () -> IO ()
threadBody name m1 m2 handle = do
  protectedPrint name m1 m2
  putMVar handle ()


main = do
  hSetBuffering stdout NoBuffering
  --Two threads, each says their name, then we join them
  handle1 <- newEmptyMVar :: IO (MVar ())
  handle2 <- newEmptyMVar :: IO (MVar ())
  mutex <- newEmptyMVar :: IO (MVar ())
  othermutex <- newEmptyMVar :: IO (MVar ())
  forkIO (threadBody "James" mutex othermutex handle1)
  forkIO (threadBody "Pritam" othermutex mutex handle2)
  takeMVar handle1
  takeMVar handle2
