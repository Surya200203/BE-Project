import React from 'react'
import { Link, NavLink } from "react-router-dom";
function Navbar() {
    return (
        <>
            <nav className='p-3 flex bg-white text-lg justify-center items-center shadow-lg fixed top-0 left-0 right-0 z-20'>
                <div id="nav-menu" className="flex gap-12 justify-end   ">

                    <NavLink to="/" className={({ isActive }) => `font-medium ${isActive ? "text-primary" : "text-gray-700"} hover:text-primary `}>
                        Home
                    </NavLink>
                    <NavLink to="/teammembers" className={({ isActive }) => `font-medium ${isActive ? "text-primary" : "text-gray-700"} hover:text-primary `}>
                        Team Members
                    </NavLink>
                    <NavLink to="/statistics" className={({ isActive }) => `font-medium ${isActive ? "text-primary" : "text-gray-700"} hover:text-primary `}>
                        Statistics
                    </NavLink>
                    <NavLink to="/methodology" className={({ isActive }) => `font-medium ${isActive ? "text-primary" : "text-gray-700"} hover:text-primary `}>
                        Methodology
                    </NavLink>


                    
                   
                </div>

            </nav>
        </>
    )
}

export default Navbar