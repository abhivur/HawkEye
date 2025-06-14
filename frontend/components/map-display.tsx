"use client"

import { useEffect, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface MapDisplayProps {
  highlightedZones: string[]
  selectedFrame: number | null
  searchResults: number[]
}

// Mock zones data
const mockZones = [
  { id: "zone-a", name: "Zone A", type: "redundant", color: "#ef4444" },
  { id: "zone-b", name: "Zone B", type: "low-coverage", color: "#f59e0b" },
  { id: "zone-c", name: "Zone C", type: "diverse", color: "#10b981" },
  { id: "zone-d", name: "Zone D", type: "optimal", color: "#3b82f6" },
]

export function MapDisplay({ highlightedZones, selectedFrame, searchResults }: MapDisplayProps) {
  const mapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // In a real implementation, you would initialize Leaflet or Mapbox here
    // For now, we'll create a simple mock map visualization
  }, [])

  return (
    <Card className="h-96 bg-zinc-900 border-zinc-800">
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-white">
          Flight Path Analysis
          <Badge variant="outline" className="border-zinc-700 text-zinc-300">
            {mockZones.length} zones identified
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          ref={mapRef}
          className="w-full h-64 bg-zinc-800 rounded-lg relative overflow-hidden border border-zinc-700"
        >
          {/* Mock map background */}
          <div className="absolute inset-0 bg-gradient-to-br from-zinc-700 to-zinc-900">
            {/* Flight path line */}
            <svg className="absolute inset-0 w-full h-full">
              <path
                d="M 20 40 Q 100 20 180 60 Q 260 100 340 80 Q 420 60 480 100"
                stroke="#ffffff"
                strokeWidth="3"
                fill="none"
                strokeDasharray="5,5"
              />
            </svg>

            {/* Zone overlays */}
            {mockZones.map((zone, index) => (
              <div
                key={zone.id}
                className={`absolute rounded-full transition-all duration-300 ${
                  highlightedZones.includes(zone.id) ? "scale-110 ring-4 ring-white" : "hover:scale-105"
                }`}
                style={{
                  left: `${20 + index * 25}%`,
                  top: `${30 + (index % 2) * 30}%`,
                  width: "60px",
                  height: "60px",
                  backgroundColor: zone.color,
                  opacity: 0.7,
                }}
              >
                <div className="absolute inset-0 flex items-center justify-center text-white text-xs font-bold">
                  {zone.name}
                </div>
              </div>
            ))}

            {/* Current frame marker */}
            {selectedFrame !== null && (
              <div
                className="absolute w-3 h-3 bg-red-500 rounded-full ring-2 ring-white animate-pulse"
                style={{
                  left: `${20 + (selectedFrame % 100) * 0.6}%`,
                  top: `${40 + Math.sin(selectedFrame * 0.1) * 20}%`,
                }}
              />
            )}

            {/* Search result markers */}
            {searchResults.map((frameId) => (
              <div
                key={frameId}
                className="absolute w-2 h-2 bg-yellow-400 rounded-full ring-1 ring-yellow-600"
                style={{
                  left: `${20 + (frameId % 100) * 0.6}%`,
                  top: `${40 + Math.sin(frameId * 0.1) * 20}%`,
                }}
              />
            ))}
          </div>
        </div>

        {/* Zone Legend */}
        <div className="mt-4 grid grid-cols-2 gap-2">
          {mockZones.map((zone) => (
            <div key={zone.id} className="flex items-center gap-2 text-sm text-zinc-300">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: zone.color }} />
              <span className="capitalize">{zone.type}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
