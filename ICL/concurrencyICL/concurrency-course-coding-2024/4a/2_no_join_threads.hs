import Control.Concurrent
import Control.Monad
import System.IO

main = do
  hSetBuffering stdout NoBuffering
  -- Two threads, each says their name
  forkIO (print "Hello I am thread 1")
  forkIO (print "Hello I am thread 2")
