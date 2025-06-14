"use client"

import { useState } from "react"
import { UploadView } from "@/components/upload-view"
import { ResultsDashboard } from "@/components/results-dashboard"

export default function HawkEyeDashboard() {
  const [currentView, setCurrentView] = useState<"upload" | "results">("upload")
  const [uploadedFiles, setUploadedFiles] = useState<{
    video: File | null
    kml: File | null
  }>({
    video: null,
    kml: null,
  })

  const handleProcessingComplete = (files: { video: File | null; kml: File | null }) => {
    setUploadedFiles(files)
    setCurrentView("results")
  }

  const handleBackToUpload = () => {
    setCurrentView("upload")
    setUploadedFiles({ video: null, kml: null })
  }

  return (
    <div className="min-h-screen bg-black">
      {currentView === "upload" ? (
        <UploadView onProcessingComplete={handleProcessingComplete} />
      ) : (
        <ResultsDashboard uploadedFiles={uploadedFiles} onBackToUpload={handleBackToUpload} />
      )}
    </div>
  )
}
