# Frontend Component Structure

This document outlines the refactored component structure for the General Agentic Memory frontend.

## Component Hierarchy

```
TransparentAIAgentUI (Main Component)
├── FileHistoryPanel (Left Sidebar)
├── ChatArea (Center)
│   ├── UploadState (Initial Upload Screen)
│   ├── MessageList
│   │   └── SourceCard (Repeated)
│   ├── InputSection
│   └── Control Buttons
└── SidePanel (Right Sidebar)
    ├── KnowledgeGraph
    ├── ThinkingSteps
    └── RetrievalDetails
```

## File Structure

### Core Components

- **`TransparentAIAgentUI.jsx`** (Main Component - 310 lines)
  - Manages all state and business logic
  - Handles API calls (upload, chat, clear memory)
  - Coordinates child components
  - Reduced from 651 lines to ~310 lines

### Layout Components

- **`FileHistoryPanel.jsx`** (Left Sidebar)
  - Displays list of uploaded files
  - Shows file metadata (size, chunks, upload time)

- **`ChatArea.jsx`** (Center Area)
  - Manages chat interface layout
  - Switches between upload state and chat state
  - Contains message list, input, and controls

- **`SidePanel.jsx`** (Right Sidebar)
  - Container for knowledge graph, thinking steps, and retrieval details
  - Organizes right-side visualizations

### Feature Components

- **`UploadState.jsx`**
  - Initial upload screen with center plus icon
  - Shows uploading state

- **`MessageList.jsx`**
  - Renders all chat messages
  - Displays retrieval details for AI messages

- **`InputSection.jsx`**
  - Message input field
  - Send button with loading states

- **`KnowledgeGraph.jsx`**
  - Canvas-based graph visualization
  - Hierarchical layout for documents, chunks, and entities
  - Beautiful gradient styling and animations

- **`ThinkingSteps.jsx`**
  - Real-time agent thinking display
  - Shows iteration steps and retrieved chunks

- **`RetrievalDetails.jsx`**
  - Displays retrieved sources with relevance scores
  - Shows source metadata

### Utility Components

- **`SourceCard.jsx`**
  - Reusable card for displaying individual sources
  - Shows document name, relevance score, snippet, and metadata

### Configuration

- **`constants.js`**
  - Centralized configuration
  - API base URL and other constants

## Benefits of This Structure

1. **Modularity**: Each component has a single, clear responsibility
2. **Reusability**: Components like `SourceCard` can be reused across the app
3. **Maintainability**: Easier to find and fix bugs in specific features
4. **Readability**: Smaller files are easier to understand
5. **Testability**: Individual components can be tested in isolation
6. **Scalability**: Easy to add new features without bloating existing files

## State Management

All state remains in the main `TransparentAIAgentUI` component and is passed down as props. This keeps the architecture simple while maintaining clear data flow.

## Styling

All components continue to use the existing `TransparentAIAgentUI.css` file. No CSS changes are required.
