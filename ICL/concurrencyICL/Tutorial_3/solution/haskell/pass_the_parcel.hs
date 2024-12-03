import Control.Concurrent
import Data.Either

type Message = Either Int String

data Logger =
  Logger (MVar ())

createLogger :: IO Logger
createLogger = do
  mutex <- newEmptyMVar
  return (Logger mutex)

loggerPrint :: Logger -> String -> IO ()
loggerPrint (Logger mutex) message = do
  putMVar mutex ()
  putStrLn message
  takeMVar mutex

music :: MVar Bool -> Logger -> IO ()
music loudspeaker logger = do
  threadDelay 300000
  takeMVar loudspeaker
  loggerPrint logger "Music off"
  putMVar loudspeaker False
  music loudspeaker logger

player ::
     String
  -> MVar Message
  -> MVar Message
  -> MVar Bool
  -> MVar String
  -> Logger
  -> IO ()
player name previousPlayer nextPlayer loudspeaker winner logger = do
  message <- takeMVar previousPlayer
  if isLeft message
      -- The player has received a parcel
      -- Get the music state
    then do
      musicState <- takeMVar loudspeaker
      threadDelay 100000
      if musicState
          -- The music is on, so just pass the parcel on.
        then do
          loggerPrint logger (name ++ ": Passing the parcel on")
          putMVar loudspeaker musicState
          putMVar nextPlayer message
          player name previousPlayer nextPlayer loudspeaker winner logger
        else do
          let (Left layerCount) = message
          let newLayerCount = layerCount - 1
          loggerPrint
            logger
            (name ++
             ": unwrapping the parcel, the layer count is now " ++ (show newLayerCount))
          -- Check whether the player has won.
          if (newLayerCount /= 0)
              -- Pass the (unwrapped) parcel on, turning the music back on in the process.
            then do
              loggerPrint logger (name ++ ": Passing the parcel on")
              loggerPrint logger "Music on"
              putMVar loudspeaker True
              putMVar nextPlayer (Left newLayerCount)
              player name previousPlayer nextPlayer loudspeaker winner logger
            else do
              loggerPrint logger (name ++ ": I am the winner!")
              -- Pass player's name to the next player to inform everyone else the player has won.
              putMVar nextPlayer (Right name)
              -- Wait to get a message back; this indicates that the winning message has made it
              -- round the ring.
              takeMVar previousPlayer
              -- Indicate to 'main' that the game is over and this player has won.
              putMVar winner name
    else do
      let (Right winner) = message
      -- Indicate that this player knows who the winner is.
      loggerPrint logger (name ++ ": The winner is " ++ winner)
      -- Pass the winning message on.
      putMVar nextPlayer message

main = do
  winner <- newEmptyMVar
  channel1to2 <- newEmptyMVar
  channel2to3 <- newEmptyMVar
  channel3to4 <- newEmptyMVar
  channel4to5 <- newEmptyMVar
  channel5to1 <- newEmptyMVar
  loudspeaker <- newMVar True
  logger <- createLogger
  forkIO (music loudspeaker logger)
  forkIO (player "Ally" channel5to1 channel1to2 loudspeaker winner logger)
  forkIO (player "Chris" channel1to2 channel2to3 loudspeaker winner logger)
  forkIO (player "Poppy" channel2to3 channel3to4 loudspeaker winner logger)
  forkIO (player "Felix" channel3to4 channel4to5 loudspeaker winner logger)
  forkIO (player "Kitty" channel4to5 channel5to1 loudspeaker winner logger)
  putMVar channel5to1 (Left 20)
  theWinner <- takeMVar winner
  loggerPrint logger ("The winner is " ++ theWinner)
