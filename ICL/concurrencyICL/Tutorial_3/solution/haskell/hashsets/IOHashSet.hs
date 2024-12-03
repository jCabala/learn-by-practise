module IOHashSet
  ( IOHashSet
  , delete
  , deleteAll
  , insert
  , insertAll
  , numBuckets
  , size
  , toList
  ) where

import Data.Hashable

class IOHashSet ioHashSet
    -- Insert an element into the hash set if not already present
  where
  insert :: (Eq a, Hashable a) => a -> ioHashSet a -> IO ()
    -- Delete an element from the hash set if present
  delete :: (Eq a, Hashable a) => a -> ioHashSet a -> IO ()
    -- Insert many elements into the hash set
  insertAll :: (Eq a, Hashable a) => [a] -> ioHashSet a -> IO ()
    -- Delete many elements from the hash set
  deleteAll :: (Eq a, Hashable a) => [a] -> ioHashSet a -> IO ()
    -- Yield the size of the hash set
  size :: ioHashSet a -> IO Int
    -- Yield the number of buckets used to represent the hash set
  numBuckets :: ioHashSet a -> IO Int
  toList :: ioHashSet a -> IO [a]
