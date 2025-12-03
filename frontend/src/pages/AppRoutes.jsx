import { Routes, Route } from 'react-router-dom'

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<div>Home Page</div>} />
      {/* Add more routes here */}
    </Routes>
  )
}

export default AppRoutes

