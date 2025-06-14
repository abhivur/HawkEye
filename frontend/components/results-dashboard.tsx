"use client"

import { useState } from "react"
import { ArrowLeft, Eye } from "lucide-react"
import { Button } from "@/components/ui/button"
import { TimelineViewer } from "@/components/timeline-viewer"
import { MapDisplay } from "@/components/map-display"
import { RecommendationsPanel } from "@/components/recommendations-panel"
import { SemanticSearch } from "@/components/semantic-search"

interface ResultsDashboardProps {
  uploadedFiles: { video: File | null; kml: File | null }
  onBackToUpload: () => void
}

export function ResultsDashboard({ uploadedFiles, onBackToUpload }: ResultsDashboardProps) {
  const [selectedFrame, setSelectedFrame] = useState<number | null>(null)
  const [highlightedZones, setHighlightedZones] = useState<string[]>([])
  const [searchResults, setSearchResults] = useState<number[]>([])

  return (
    <div className="min-h-screen bg-black">
      {/* Header */}
      <div className="bg-zinc-900 border-b border-zinc-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={onBackToUpload}
              className="flex items-center gap-2 text-zinc-300 hover:text-white hover:bg-zinc-800"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Upload
            </Button>
            <div className="flex items-center gap-2">
              <Eye className="h-6 w-6 text-white" />
              <h1 className="text-xl font-bold text-white">HawkEye Dashboard</h1>
            </div>
          </div>
          <div className="text-sm text-zinc-400">
            Video: {uploadedFiles.video?.name} | Flight Plan: {uploadedFiles.kml?.name}
          </div>
        </div>
      </div>

      {/* Search Bar */}
      <div className="bg-zinc-900 border-b border-zinc-800 px-6 py-3">
        <SemanticSearch onSearchResults={setSearchResults} onClearSearch={() => setSearchResults([])} />
      </div>

      {/* Main Dashboard */}
      <div className="flex-1 grid lg:grid-cols-3 gap-6 p-6 bg-black">
        {/* Left Column - Timeline and Video */}
        <div className="lg:col-span-2 space-y-6">
          <TimelineViewer
            selectedFrame={selectedFrame}
            onFrameSelect={setSelectedFrame}
            searchResults={searchResults}
          />
        </div>

        {/* Right Column - Map and Recommendations */}
        <div className="space-y-6">
          <MapDisplay highlightedZones={highlightedZones} selectedFrame={selectedFrame} searchResults={searchResults} />
          <RecommendationsPanel onZoneHover={setHighlightedZones} onZoneLeave={() => setHighlightedZones([])} />
        </div>
      </div>
    </div>
  )
}
