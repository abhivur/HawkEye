"use client"

import { useState, useRef, useEffect } from "react"
import { Play, Pause, SkipBack, SkipForward, Volume2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"

interface TimelineViewerProps {
  selectedFrame: number | null
  onFrameSelect: (frame: number) => void
  searchResults: number[]
}

// Mock data for demonstration
const mockFrames = Array.from({ length: 120 }, (_, i) => ({
  id: i,
  timestamp: `${Math.floor(i / 60)}:${(i % 60).toString().padStart(2, "0")}`,
  thumbnail: `/placeholder.svg?height=60&width=80&text=Frame${i + 1}`,
  caption: [
    "Vehicle near fence",
    "Person walking",
    "White truck parked",
    "Building structure",
    "Open field area",
    "Tree line visible",
    "Road intersection",
    "Construction site",
    "Parking lot",
    "Residential area",
  ][i % 10],
  hasActivity: Math.random() > 0.7,
}))

export function TimelineViewer({ selectedFrame, onFrameSelect, searchResults }: TimelineViewerProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [volume, setVolume] = useState([80])
  const timelineRef = useRef<HTMLDivElement>(null)

  const duration = mockFrames.length

  useEffect(() => {
    if (selectedFrame !== null) {
      setCurrentTime(selectedFrame)
      setIsPlaying(false)
    }
  }, [selectedFrame])

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleTimeChange = (value: number[]) => {
    setCurrentTime(value[0])
    onFrameSelect(value[0])
  }

  const scrollToFrame = (frameId: number) => {
    const frameElement = document.getElementById(`frame-${frameId}`)
    if (frameElement && timelineRef.current) {
      frameElement.scrollIntoView({ behavior: "smooth", inline: "center" })
    }
  }

  return (
    <Card className="h-full bg-zinc-900 border-zinc-800">
      <CardHeader>
        <CardTitle className="text-white">Video Timeline</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Video Player Placeholder */}
        <div className="aspect-video bg-black rounded-lg flex items-center justify-center border border-zinc-800">
          <div className="text-white text-center">
            <div className="text-6xl mb-2">📹</div>
            <p>Video Player</p>
            <p className="text-sm opacity-75">
              Frame {currentTime + 1} of {duration}
            </p>
          </div>
        </div>

        {/* Video Controls */}
        <div className="space-y-3">
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleTimeChange([Math.max(0, currentTime - 10)])}
              className="border-zinc-700 text-white hover:bg-zinc-800"
            >
              <SkipBack className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handlePlayPause}
              className="border-zinc-700 text-white hover:bg-zinc-800"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleTimeChange([Math.min(duration - 1, currentTime + 10)])}
              className="border-zinc-700 text-white hover:bg-zinc-800"
            >
              <SkipForward className="h-4 w-4" />
            </Button>
            <div className="flex items-center gap-2 ml-auto">
              <Volume2 className="h-4 w-4 text-white" />
              <Slider value={volume} onValueChange={setVolume} max={100} step={1} className="w-20" />
            </div>
          </div>

          {/* Progress Bar */}
          <Slider
            value={[currentTime]}
            onValueChange={handleTimeChange}
            max={duration - 1}
            step={1}
            className="w-full"
          />
        </div>

        {/* Frame Timeline */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-white">Frame Timeline</h3>
            {searchResults.length > 0 && (
              <Badge variant="secondary" className="bg-zinc-800 text-white">
                {searchResults.length} matches found
              </Badge>
            )}
          </div>
          <div
            ref={timelineRef}
            className="flex gap-2 overflow-x-auto pb-2 max-h-32"
            style={{ scrollbarWidth: "thin" }}
          >
            {mockFrames.map((frame) => (
              <div
                key={frame.id}
                id={`frame-${frame.id}`}
                className={`flex-shrink-0 cursor-pointer transition-all ${
                  currentTime === frame.id
                    ? "ring-2 ring-white scale-105"
                    : searchResults.includes(frame.id)
                      ? "ring-2 ring-yellow-400"
                      : "hover:scale-105"
                }`}
                onClick={() => {
                  onFrameSelect(frame.id)
                  scrollToFrame(frame.id)
                }}
              >
                <div className="relative">
                  <img
                    src={frame.thumbnail || "/placeholder.svg"}
                    alt={`Frame ${frame.id + 1}`}
                    className="w-20 h-15 object-cover rounded border border-zinc-700"
                  />
                  {frame.hasActivity && <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></div>}
                  <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-75 text-white text-xs p-1 rounded-b">
                    {frame.timestamp}
                  </div>
                </div>
                <div className="mt-1 text-xs text-center max-w-20 truncate text-zinc-300">{frame.caption}</div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
