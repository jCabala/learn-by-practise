import Control.Concurrent
import Control.Monad
import System.IO

putChars :: Char -> Int -> IO ()
putChars c 1 = putChar c
putChars c n = do
  putChar c
  putChars c (n - 1)

main = do
  hSetBuffering stdout NoBuffering
  -- Launch thread that puts lots of Bs
  forkIO (putChars 'B' 1000)
  putChars 'A' 1000
