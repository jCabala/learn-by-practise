import Control.Concurrent

import qualified CoarseGrainedHashSet
import IOHashSet
import qualified StripedHashSet

import Control.Monad
import Data.List
import Formatting
import Formatting.Clock
import System.Clock
import System.Environment
import Text.Printf

thread :: IOHashSet hashSet => hashSet String -> Int -> Int -> MVar () -> IO ()
thread iohs start end handle = do
  IOHashSet.insertAll (map show [start .. end]) iohs
  IOHashSet.deleteAll (map show [x | x <- [start .. end], even x]) iohs
  putMVar handle ()

benchmark :: IOHashSet hashSet => hashSet String -> Int -> Int -> IO ()
benchmark hs nthreads chunkSize = do
  computationStart <- getTime Realtime
  handles <- mapM (\x -> newEmptyMVar) [0 .. nthreads - 1]
  mapM_
    (\x ->
       forkIO
         (thread
            hs
            (chunkSize * (snd x))
            (chunkSize * ((snd x) + 1) - 1)
            (fst x)))
    (zip handles [0 .. nthreads - 1])
  mapM_ takeMVar handles
  computationEnd <- getTime Realtime
  checkingStart <- getTime Realtime
  sz <- IOHashSet.size hs
  let expectedSize = ((nthreads * chunkSize) `div` 2)
  when
    (sz /= expectedSize)
    (error ("Bad size " ++ (show sz) ++ ", expected " ++ (show expectedSize)))
  content <- IOHashSet.toList hs
  let expectedContent = map show [1,3 .. (nthreads * chunkSize)]
  when ((sort content) /= (sort expectedContent)) (error "Unexpected result")
  checkingEnd <- getTime Realtime
  print "Success"
  printf "Concurrent computation time: "
  fprint timeSpecs computationStart computationEnd
  printf "\nChecking time:               "
  fprint timeSpecs checkingStart checkingEnd
  printf "\n"

main = do
  args <- getArgs
  when
    ((length args) /= 3)
    (error "Usage: benchmark nthreads chunkSize hashSetKind")
  let nthreads = read (args !! 0) :: Int
  let chunkSize = read (args !! 1) :: Int
  let hashSetKind = args !! 2
  if hashSetKind == "coarse"
    then do
      hs <- (CoarseGrainedHashSet.coarseGrainedHashSet 16)
      benchmark hs nthreads chunkSize
    else do
      if hashSetKind == "striped"
        then do
          hs <- (StripedHashSet.stripedHashSet 16)
          benchmark hs nthreads chunkSize
        else error
               ("Unknown hash set kind: " ++
                hashSetKind ++ ", options are 'coarse' and 'striped'")
