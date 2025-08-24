import { useState } from 'react'
import './App.css'
import ImageDeblurring from './components/ImageDeblurring'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div className = "App">
      <ImageDeblurring />
    </div>
  )
}

export default App
