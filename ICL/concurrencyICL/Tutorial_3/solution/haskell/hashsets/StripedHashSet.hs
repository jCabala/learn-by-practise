module StripedHashSet
  ( StripedHashSet
  , delete
  , deleteAll
  , insert
  , insertAll
  , numBuckets
  , size
  , stripedHashSet
  , toList
  ) where

import Control.Concurrent
import Control.Monad
import Data.Array
import Data.Atomics.Counter
import Data.Hashable

import Data.IORef
import qualified Data.List

import IOHashSet

-- A striped hash set comprises:
-- * An integer-indexed array of mutexes (unit MVars), whose size does not
--   change.
-- * An integer-indexed array of mutable buckets, whose size does change. Each
--   bucket is represented as a list MVar. The array of buckets is held in an
--   IORef so that it can be updated when a resize occurs.
-- * A number of buckets, which can change, hence is stored in an IORef
-- * A size, which can change, hence is stored in an MVar.
data StripedHashSet a =
  StripedHashSet
    (Array Int (MVar ())) -- Fixed array of mutexes to protect the buckets
    (IORef (Array Int (MVar [a]))) -- Array of buckets, which can be updated
    (IORef Int) -- Number of buckets
    AtomicCounter -- Size of the hash set

instance IOHashSet StripedHashSet where
  insert x (StripedHashSet mutexes bucketsRef numBucketsRef sizeCounter) = do
    let shs = StripedHashSet mutexes bucketsRef numBucketsRef sizeCounter
    -- Lock the relevant bucket
    let mutexIndex = hash x `mod` snd (bounds mutexes)
    takeMVar (mutexes ! mutexIndex)
    -- Get the bucket based on the current bucket count. The bucket count
    -- will not change while we hold a bucket mutex, because doing a resize
    -- requires holding *all* bucket mutexes.
    numBuckets <- numBuckets shs
    buckets <- readIORef bucketsRef
    let bucketIndex = hash x `mod` numBuckets
    bucket <- takeMVar (buckets ! bucketIndex)
    let insertNeeded = x `notElem` bucket
    -- Put the bucket back, with the new element added if necessary.
    let bucket' =
          (if insertNeeded
             then x : bucket
             else bucket)
    putMVar (buckets ! bucketIndex) bucket'
    when
      insertNeeded
        -- An element was inserted, so increase the hash set size.
      (do incrCounter 1 sizeCounter
          return ())
    -- Unlock the bucket.
    putMVar (mutexes ! mutexIndex) ()
    -- Evaluate the bucket to avoid accumulation of unevaluated thunks; do this
    -- after the mutex has been released to avoid a lengthy critical section.
    seq bucket' (return ())
    -- Rsize the hash set if it has gotten too big.
    resizeNeeded <- policy shs
    when resizeNeeded (resize shs)
  delete x (StripedHashSet mutexes bucketsRef numBucketsRef sizeCounter) = do
    let shs = StripedHashSet mutexes bucketsRef numBucketsRef sizeCounter
    let mutexIndex = hash x `mod` snd (bounds mutexes)
    takeMVar (mutexes ! mutexIndex)
    buckets <- readIORef bucketsRef
    bucketCount <- numBuckets shs
    let bucketIndex = hash x `mod` bucketCount
    bucket <- takeMVar (buckets ! bucketIndex)
    let deleteNeeded = x `elem` bucket
    let bucket' =
          (if deleteNeeded
             then Data.List.delete x bucket
             else bucket)
    putMVar (buckets ! bucketIndex) bucket'
    when
      deleteNeeded
      (do incrCounter (-1) sizeCounter
          return ())
    putMVar (mutexes ! mutexIndex) ()
    seq bucket' (return ())
  insertAll [] shs = return ()
  insertAll (x:xs) shs = do
    insert x shs
    insertAll xs shs
  deleteAll [] shs = return ()
  deleteAll (x:xs) shs = do
    delete x shs
    deleteAll xs shs
  numBuckets (StripedHashSet _ _ numBucketsRef _) = readIORef numBucketsRef
  toList shs = do
    acquireAllMutexes shs
    content <- toListUnsynchronized shs
    releaseAllMutexes shs
    return content
  size (StripedHashSet _ _ _ sizeCounter) = readCounter sizeCounter

-- Create a hash set with the given number of buckets
stripedHashSet :: Int -> IO (StripedHashSet a)
stripedHashSet bucketCount
  -- Create |bucketCount| unit MVars - these are the mutexes
 = do
  mutexes <- mapM (\x -> newMVar () :: IO (MVar ())) [1 .. bucketCount]
  -- The mutex array is an array of these MVars
  let range = (0, bucketCount - 1)
  let mutexArray = array range (zip [0 .. bucketCount - 1] mutexes)
  -- Create |capcity| list MVars, each holding empty lists - these are the
  -- buckets
  buckets <- mapM (\x -> newMVar [] :: IO (MVar [a])) [1 .. bucketCount]
  -- The buckets array is an array of these MVars, itself stored in an MVar so
  -- that it can be changed when the hash set is resized
  bucketsRef <- newIORef (array range (zip [0 .. bucketCount - 1] buckets))
  -- MVars for the number of buckets and the size
  numBucketsRef <- newIORef bucketCount
  sizeCounter <- newCounter 0
  return (StripedHashSet mutexArray bucketsRef numBucketsRef sizeCounter)

-- Determine whether a resize is needed
policy :: StripedHashSet a -> IO Bool
policy hs = do
  currentSize <- size hs
  currentNumBuckets <- numBuckets hs
  return (currentSize `div` currentNumBuckets > 4)

acquireAllMutexes :: StripedHashSet a -> IO ()
acquireAllMutexes (StripedHashSet mutexes _ _ _) = do
  mapM_ (\index -> takeMVar (mutexes ! index)) [0 .. snd (bounds mutexes) - 1]

releaseAllMutexes :: StripedHashSet a -> IO ()
releaseAllMutexes (StripedHashSet mutexes _ _ _) = do
  mapM_ (\index -> putMVar (mutexes ! index) ()) [0 .. snd (bounds mutexes) - 1]

resize :: (Eq a, Hashable a) => StripedHashSet a -> IO ()
resize (StripedHashSet mutexes bucketsRef bucketCountRef sizeCounter) = do
  let shs = StripedHashSet mutexes bucketsRef bucketCountRef sizeCounter
  -- Acquire all the bucket mutexes.
  acquireAllMutexes shs
  -- Check whether a resize is still needed - it could bre that another thread
  -- took care of the resize while we were acquiring the mutexes.
  stillNeedToResize <- policy shs
  when
    stillNeedToResize
      -- A resize is still needed, so:
      -- Get the existing content as a list
    (do content <- toListUnsynchronized shs
      -- Update the bucket count of the hash set
        oldBucketCount <- readIORef bucketCountRef
        let newBucketCount = oldBucketCount * 2
        writeIORef bucketCountRef newBucketCount
      -- Make a list of new MVars, one for each new bucket, putting the
      -- appropriate old bucket contents into it
        newBuckets <-
          mapM
            (\index ->
               newMVar
                 [x | x <- content, (hash x `mod` newBucketCount) == index])
            [0 .. newBucketCount - 1]
      -- Store the new buckets to the hash set's bucket array MVar
        writeIORef bucketsRef $!
          array
            (0, newBucketCount - 1)
            (zip [0 .. newBucketCount - 1] newBuckets))
  releaseAllMutexes shs

toListUnsynchronized :: StripedHashSet a -> IO [a]
toListUnsynchronized shs = do
  let (StripedHashSet _ bucketsRef _ _) = shs
  bucketCount <- numBuckets shs
  buckets <- readIORef bucketsRef
  content <- mapM (\index -> readMVar (buckets ! index)) [0 .. bucketCount - 1]
  return (concat content)
