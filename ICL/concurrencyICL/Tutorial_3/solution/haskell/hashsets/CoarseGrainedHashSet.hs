module CoarseGrainedHashSet
  ( CoarseGrainedHashSet
  , coarseGrainedHashSet
  , delete
  , deleteAll
  , insert
  , insertAll
  , numBuckets
  , size
  , toList
  ) where

import Control.Concurrent

-- This allows you to refer to functions from the (non-concurrent) HashSet module, but in a
-- *qualified* fashion; e.g. you write HashSet.insert to refer to the insert function on a
-- non-concurrent hash set
import qualified HashSet

-- This non-qualified import means that when you refer to `insert` or `insertAll`, without
-- qualification, this will refer to functions from IOHashSet
import IOHashSet

newtype CoarseGrainedHashSet a =
  CoarseGrainedHashSet (MVar (HashSet.HashSet a))

instance IOHashSet CoarseGrainedHashSet where
  insert x (CoarseGrainedHashSet m) = do
    hs <- takeMVar m
    let temp = HashSet.insert x hs
    putMVar m temp
    seq temp (return ())
  insertAll [] cghs = return ()
  insertAll (x:xs) cghs = do
    insert x cghs
    insertAll xs cghs
  delete x (CoarseGrainedHashSet m) = do
    hs <- takeMVar m
    let temp = HashSet.delete x hs
    putMVar m temp
    seq temp (return ())
  deleteAll [] cghs = return ()
  deleteAll (x:xs) cghs = do
    delete x cghs
    deleteAll xs cghs
  size (CoarseGrainedHashSet m) = do
    hs <- readMVar m
    return (HashSet.size hs)
  numBuckets (CoarseGrainedHashSet m) = do
    hs <- readMVar m
    return (HashSet.numBuckets hs)
  toList (CoarseGrainedHashSet m) = do
    hs <- readMVar m
    return (HashSet.toList hs)

-- This function should create an empty CoarseGrainedHashSet with the given number of initial
-- buckets.
coarseGrainedHashSet :: Int -> IO (CoarseGrainedHashSet a)
coarseGrainedHashSet bucketCount = do
  set <- newMVar (HashSet.hashSet bucketCount)
  return (CoarseGrainedHashSet set)
