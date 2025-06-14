"use client"

import type React from "react"

import { useState } from "react"
import { Search, X } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface SemanticSearchProps {
  onSearchResults: (results: number[]) => void
  onClearSearch: () => void
}

const mockSearchData = {
  "person walking": [15, 23, 45, 67, 89],
  "white truck": [8, 34, 56, 78, 91],
  "vehicle near fence": [12, 28, 41, 63, 85],
  "building structure": [5, 19, 37, 52, 74],
  "construction site": [22, 38, 49, 71, 93],
  "parking lot": [11, 29, 44, 66, 88],
  "road intersection": [7, 25, 43, 59, 81],
  "tree line": [14, 31, 47, 69, 95],
}

export function SemanticSearch({ onSearchResults, onClearSearch }: SemanticSearchProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [isSearching, setIsSearching] = useState(false)
  const [currentResults, setCurrentResults] = useState<number[]>([])

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setIsSearching(true)

    // Simulate search delay
    await new Promise((resolve) => setTimeout(resolve, 800))

    // Mock search logic - find matching scenes
    const query = searchQuery.toLowerCase()
    let results: number[] = []

    Object.entries(mockSearchData).forEach(([key, frames]) => {
      if (key.includes(query) || query.includes(key.split(" ")[0])) {
        results = [...results, ...frames]
      }
    })

    // Remove duplicates and sort
    results = [...new Set(results)].sort((a, b) => a - b)

    setCurrentResults(results)
    onSearchResults(results)
    setIsSearching(false)
  }

  const handleClear = () => {
    setSearchQuery("")
    setCurrentResults([])
    onClearSearch()
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch()
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-zinc-400" />
          <Input
            type="text"
            placeholder="Search for scenes (e.g., 'person walking', 'white truck', 'building structure')"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            className="pl-10 pr-10 bg-zinc-800 border-zinc-700 text-white placeholder-zinc-400"
            disabled={isSearching}
          />
          {searchQuery && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClear}
              className="absolute right-1 top-1/2 transform -translate-y-1/2 h-6 w-6 p-0 text-zinc-400 hover:text-white"
            >
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
        <Button
          onClick={handleSearch}
          disabled={!searchQuery.trim() || isSearching}
          className="px-6 bg-white text-black hover:bg-zinc-200"
        >
          {isSearching ? "Searching..." : "Search"}
        </Button>
      </div>

      {currentResults.length > 0 && (
        <div className="mt-3 flex items-center gap-2">
          <Badge variant="secondary" className="bg-zinc-800 text-white">
            {currentResults.length} frames found
          </Badge>
          <span className="text-sm text-zinc-400">for "{searchQuery}"</span>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClear}
            className="ml-auto text-xs text-zinc-400 hover:text-white"
          >
            Clear results
          </Button>
        </div>
      )}

      {/* Search suggestions */}
      {!searchQuery && (
        <div className="mt-3">
          <p className="text-sm text-zinc-400 mb-2">Try searching for:</p>
          <div className="flex flex-wrap gap-2">
            {Object.keys(mockSearchData)
              .slice(0, 6)
              .map((suggestion) => (
                <Badge
                  key={suggestion}
                  variant="outline"
                  className="cursor-pointer hover:bg-zinc-800 border-zinc-700 text-zinc-300"
                  onClick={() => setSearchQuery(suggestion)}
                >
                  {suggestion}
                </Badge>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}
