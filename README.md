# LangGraph SDK Documentation

This documentation explains the concepts and usage of the `@langchain/langgraph-sdk` package. It provides an overview of all classes, methods, and properties, along with examples and their integration with a Next.js frontend using the App Router. A real-world example is included at the end.

---

## Installation

To use the LangGraph SDK, install it via npm or yarn:

```bash
npm install @langchain/langgraph-sdk
```

---

## Classes

### **AssistantsClient**

This client handles operations related to creating, managing, and retrieving assistants.

#### **Constructor**

```typescript
const assistantsClient = new AssistantsClient(config);
```
- **`config`** *(optional)*: `ClientConfig` object for configuration.

#### **Methods**

##### `create(payload)`
Creates a new assistant.

**Parameters**:
- `payload`:
  - `assistantId?`: *(optional)* ID of the assistant.
  - `config?`: *(optional)* Configuration for the assistant.
  - `graphId`: ID of the graph.
  - `ifExists?`: *(optional)* Behavior if assistant already exists.
  - `metadata?`: *(optional)* Metadata.
  - `name?`: *(optional)* Name of the assistant.

**Returns**: `Promise<Assistant>`

**Example**:

```typescript
const assistant = await assistantsClient.create({
  graphId: "graph-id-123",
  name: "My Assistant",
});
console.log("Created Assistant:", assistant);
```

##### `delete(assistantId)`
Deletes an assistant by ID.

**Parameters**:
- `assistantId`: ID of the assistant.

**Returns**: `Promise<void>`

**Example**:

```typescript
await assistantsClient.delete("assistant-id-123");
console.log("Assistant deleted.");
```

##### `get(assistantId)`
Retrieves an assistant by ID.

**Parameters**:
- `assistantId`: ID of the assistant.

**Returns**: `Promise<Assistant>`

**Example**:

```typescript
const assistant = await assistantsClient.get("assistant-id-123");
console.log("Assistant Details:", assistant);
```

---

### **Client**

The `Client` class acts as the main entry point for the LangGraph SDK, allowing access to sub-clients like Assistants, Threads, Store, Crons, and Runs.

#### **Constructor**

```typescript
const client = new Client(config);
```
- **`config`** *(optional)*: `ClientConfig` object for configuration.

#### **Properties**

- **`assistants`**: Instance of `AssistantsClient`.
- **`threads`**: Instance of `ThreadsClient`.
- **`store`**: Instance of `StoreClient`.
- **`runs`**: Instance of `RunsClient`.
- **`crons`**: Instance of `CronsClient`.

---

### **ThreadsClient**

Handles operations related to threads.

#### **Methods**

##### `create(payload)`
Creates a new thread.

**Parameters**:
- `payload`: Configuration for creating a thread.

**Returns**: `Promise<Thread>`

**Example**:

```typescript
const thread = await client.threads.create({
  metadata: { purpose: "example-thread" },
});
console.log("Created Thread:", thread);
```

##### `get(threadId)`
Retrieves a thread by ID.

**Parameters**:
- `threadId`: ID of the thread.

**Returns**: `Promise<Thread>`

**Example**:

```typescript
const thread = await client.threads.get("thread-id-123");
console.log("Thread Details:", thread);
```

---

## Example: Integrating with Next.js (App Router)

### Setting Up
1. Install dependencies:

```bash
npm install @langchain/langgraph-sdk
```

2. Create an API route in your Next.js App Router (`/app/api/assistants/route.ts`):

```typescript
import { AssistantsClient } from "@langchain/langgraph-sdk";

const assistantsClient = new AssistantsClient();

export async function POST(request: Request) {
  const body = await request.json();
  const { name, graphId } = body;

  try {
    const assistant = await assistantsClient.create({
      name,
      graphId,
    });
    return new Response(JSON.stringify(assistant), { status: 201 });
  } catch (error) {
    console.error("Error creating assistant:", error);
    return new Response("Failed to create assistant", { status: 500 });
  }
}
```

### Frontend Example
Create a form to create an assistant:

```typescript
'use client';

import { useState } from "react";

export default function CreateAssistant() {
  const [name, setName] = useState("");
  const [graphId, setGraphId] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const res = await fetch("/api/assistants", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name, graphId }),
    });

    const data = await res.json();
    console.log("Created Assistant:", data);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Assistant Name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <input
        type="text"
        placeholder="Graph ID"
        value={graphId}
        onChange={(e) => setGraphId(e.target.value)}
      />
      <button type="submit">Create Assistant</button>
    </form>
  );
}
```

---

## Real-World Example: Automated Task Management

Use LangGraph SDK to manage tasks, create threads, and utilize assistants for automating workflows.

### Backend

1. **Create a task automation graph**:

```typescript
import { Client } from "@langchain/langgraph-sdk";

const client = new Client();

async function createTaskAutomation() {
  const assistant = await client.assistants.create({
    graphId: "task-graph-id",
    name: "Task Automation Assistant",
  });

  console.log("Assistant Created:", assistant);
}

createTaskAutomation();
```

2. **Trigger runs for tasks**:

```typescript
async function runTask(assistantId, threadId) {
  const run = await client.runs.create(threadId, assistantId, {
    payload: {
      task: "Automate daily report generation",
    },
  });

  console.log("Run Created:", run);
}
```

### Frontend

Display ongoing tasks and statuses using the ThreadsClient:

```typescript
import { useEffect, useState } from "react";

export default function TaskManager() {
  const [tasks, setTasks] = useState([]);

  useEffect(() => {
    async function fetchTasks() {
      const res = await fetch("/api/threads");
      const data = await res.json();
      setTasks(data);
    }

    fetchTasks();
  }, []);

  return (
    <div>
      <h1>Task Manager</h1>
      <ul>
        {tasks.map((task) => (
          <li key={task.id}>{task.metadata.purpose}</li>
        ))}
      </ul>
    </div>
  );
}
```

---

With this documentation, you can easily understand and utilize the LangGraph SDK to build robust applications integrated with Next.js.
