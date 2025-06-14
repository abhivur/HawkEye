"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Upload, FileVideo, Map, Loader2, Eye } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

interface UploadViewProps {
  onProcessingComplete: (files: { video: File | null; kml: File | null }) => void
}

export function UploadView({ onProcessingComplete }: UploadViewProps) {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [kmlFile, setKmlFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [processingStage, setProcessingStage] = useState("")

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>, fileType: "video" | "kml") => {
    const file = event.target.files?.[0]
    if (file) {
      if (fileType === "video") {
        setVideoFile(file)
      } else {
        setKmlFile(file)
      }
    }
  }, [])

  const handleSubmit = async () => {
    setIsProcessing(true)
    setProcessingProgress(0)

    // Simulate processing stages
    const stages = [
      "Analyzing video frames...",
      "Processing flight path data...",
      "Generating AI captions...",
      "Identifying optimization zones...",
      "Creating recommendations...",
      "Finalizing results...",
    ]

    for (let i = 0; i < stages.length; i++) {
      setProcessingStage(stages[i])
      setProcessingProgress((i + 1) * (100 / stages.length))
      await new Promise((resolve) => setTimeout(resolve, 1500))
    }

    onProcessingComplete({ video: videoFile, kml: kmlFile })
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  if (isProcessing) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4 bg-black">
        <Card className="w-full max-w-md bg-zinc-900 border-zinc-800">
          <CardHeader className="text-center">
            <div className="mx-auto mb-4 p-3 bg-white rounded-full w-fit">
              <Loader2 className="h-8 w-8 text-black animate-spin" />
            </div>
            <CardTitle className="text-white">Processing Your Data</CardTitle>
            <CardDescription className="text-zinc-400">Analyzing drone footage and flight path data</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-zinc-300">
                <span>{processingStage}</span>
                <span>{Math.round(processingProgress)}%</span>
              </div>
              <Progress value={processingProgress} className="h-2" />
            </div>
            <p className="text-sm text-zinc-400 text-center">This may take a few minutes depending on video length</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen p-4 bg-black">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Eye className="h-8 w-8 text-white" />
            <h1 className="text-3xl font-bold text-white">HawkEye</h1>
          </div>
          <p className="text-lg text-zinc-400">Optimize your drone footage with AI-powered analysis</p>
        </div>

        {/* Upload Cards */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Video Upload */}
          <Card className="border-2 border-dashed border-zinc-700 hover:border-white transition-colors bg-zinc-900">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <FileVideo className="h-5 w-5 text-white" />
                Drone Video
              </CardTitle>
              <CardDescription className="text-zinc-400">Upload your drone footage (.mp4 format)</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <label className="block">
                  <input
                    type="file"
                    accept=".mp4,video/mp4"
                    onChange={(e) => handleFileUpload(e, "video")}
                    className="hidden"
                  />
                  <div className="cursor-pointer p-6 text-center border-2 border-dashed border-zinc-700 rounded-lg hover:border-white hover:bg-zinc-800 transition-colors">
                    <Upload className="h-8 w-8 text-zinc-400 mx-auto mb-2" />
                    <p className="text-sm text-zinc-400">Click to upload or drag and drop</p>
                  </div>
                </label>
                {videoFile && (
                  <div className="p-3 bg-zinc-800 border border-zinc-600 rounded-lg">
                    <p className="text-sm font-medium text-white">{videoFile.name}</p>
                    <p className="text-xs text-zinc-400">{formatFileSize(videoFile.size)}</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* KML Upload */}
          <Card className="border-2 border-dashed border-zinc-700 hover:border-white transition-colors bg-zinc-900">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Map className="h-5 w-5 text-white" />
                Flight Plan (KML)
              </CardTitle>
              <CardDescription className="text-zinc-400">
                Upload your flight path from Google Earth (.kml format)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <label className="block">
                  <input
                    type="file"
                    accept=".kml,application/vnd.google-earth.kml+xml"
                    onChange={(e) => handleFileUpload(e, "kml")}
                    className="hidden"
                  />
                  <div className="cursor-pointer p-6 text-center border-2 border-dashed border-zinc-700 rounded-lg hover:border-white hover:bg-zinc-800 transition-colors">
                    <Upload className="h-8 w-8 text-zinc-400 mx-auto mb-2" />
                    <p className="text-sm text-zinc-400">Click to upload or drag and drop</p>
                  </div>
                </label>
                {kmlFile && (
                  <div className="p-3 bg-zinc-800 border border-zinc-600 rounded-lg">
                    <p className="text-sm font-medium text-white">{kmlFile.name}</p>
                    <p className="text-xs text-zinc-400">{formatFileSize(kmlFile.size)}</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Submit Button */}
        <div className="text-center">
          <Button onClick={handleSubmit} size="lg" className="px-8 py-3 text-lg bg-white text-black hover:bg-zinc-200">
            Submit & Process
          </Button>
          <p className="text-sm text-zinc-400 mt-2">
            {!videoFile || !kmlFile ? "Demo mode - processing with sample data" : "Ready to process your files"}
          </p>
        </div>
      </div>
    </div>
  )
}
