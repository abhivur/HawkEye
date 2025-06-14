"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertTriangle, TrendingUp, Target, MapPin } from "lucide-react"

interface RecommendationsPanelProps {
  onZoneHover: (zones: string[]) => void
  onZoneLeave: () => void
}

const mockRecommendations = [
  {
    id: 1,
    type: "altitude",
    priority: "high",
    title: "Fly lower over Zone C",
    description:
      "Current altitude of 120m is too high for detailed ground analysis. Recommend 80m for better resolution.",
    relatedZones: ["zone-c"],
    icon: TrendingUp,
  },
  {
    id: 2,
    type: "coverage",
    priority: "medium",
    title: "Increase pass frequency in Zone D",
    description:
      "Area shows high activity but limited coverage. Add 2-3 additional passes for comprehensive monitoring.",
    relatedZones: ["zone-d"],
    icon: Target,
  },
  {
    id: 3,
    type: "efficiency",
    priority: "low",
    title: "Avoid duplicating path over Zone B",
    description: "Flight path shows 40% overlap in this area. Optimize route to reduce redundancy and save battery.",
    relatedZones: ["zone-b"],
    icon: AlertTriangle,
  },
  {
    id: 4,
    type: "positioning",
    priority: "medium",
    title: "Adjust angle for Zone A",
    description:
      "Current camera angle creates shadows. Fly during different time or adjust gimbal for better visibility.",
    relatedZones: ["zone-a"],
    icon: MapPin,
  },
]

const priorityColors = {
  high: "bg-red-900/30 text-red-300 border-red-700",
  medium: "bg-yellow-900/30 text-yellow-300 border-yellow-700",
  low: "bg-green-900/30 text-green-300 border-green-700",
}

export function RecommendationsPanel({ onZoneHover, onZoneLeave }: RecommendationsPanelProps) {
  return (
    <Card className="h-96 bg-zinc-900 border-zinc-800">
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-white">
          Flight Recommendations
          <Badge variant="outline" className="border-zinc-700 text-zinc-300">
            {mockRecommendations.length} suggestions
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3 max-h-80 overflow-y-auto">
          {mockRecommendations.map((rec) => {
            const IconComponent = rec.icon
            return (
              <div
                key={rec.id}
                className="p-3 border border-zinc-700 rounded-lg hover:bg-zinc-800 transition-colors cursor-pointer"
                onMouseEnter={() => onZoneHover(rec.relatedZones)}
                onMouseLeave={onZoneLeave}
              >
                <div className="flex items-start gap-3">
                  <div className="p-1 bg-zinc-800 rounded">
                    <IconComponent className="h-4 w-4 text-white" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-medium text-sm text-white">{rec.title}</h4>
                      <Badge variant="outline" className={`text-xs ${priorityColors[rec.priority]}`}>
                        {rec.priority}
                      </Badge>
                    </div>
                    <p className="text-xs text-zinc-400 leading-relaxed">{rec.description}</p>
                    <div className="flex items-center gap-1 mt-2">
                      <MapPin className="h-3 w-3 text-zinc-500" />
                      <span className="text-xs text-zinc-500">{rec.relatedZones.join(", ").toUpperCase()}</span>
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
