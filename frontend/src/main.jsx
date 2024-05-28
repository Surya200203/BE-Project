import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import Hero from './components/Hero.jsx'
import Layout from './components/Layout.jsx'
import TeamMember from './components/TeamMember.jsx'

import { RouterProvider, createBrowserRouter } from 'react-router-dom'
import Statistics from './components/Statistics.jsx'
import Methodology from './components/Methodology.jsx'
import ImageUpload from './components/ImageUpload.jsx'



const router = createBrowserRouter([
  {
    path: '/',
    element:<Layout/>,
    children:[
      {
        path:"",
        element:<Hero/>
      },
      {
        path:"",
        element:<ImageUpload/>
      },
      {
        path:"/teammembers",
        element: <TeamMember />
      },
      {
        path:"/statistics",
        element:<Statistics/>
      },
      {
        path:"/methodology",
        element: <Methodology />
      }
    ]
  }
])

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {/* <App /> */}
    <RouterProvider router={router}/>
  </React.StrictMode>,
)
