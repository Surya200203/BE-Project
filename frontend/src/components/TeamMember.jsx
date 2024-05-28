import React from 'react'
import { Link, NavLink } from "react-router-dom";
function TeamMember() {
    return (
        <>

            <div className='container  flex  gap-8 justify-around mt-8 mr-10 ml-10'>
                <div id="left-part" className='flex items-center'>
                    <h2 className='text-5xl max-w-sm sm:font-semibold mb-14 leading-snug'>Meet Our Team Members</h2>
                </div>
                <div id='right-part' className='grid grid-cols-2 gap-8  justify-center items-center'>

                    <div id='card' className='max-w-xl  p-4 flex flex-col gap-2 items-center justify-center'>
                        <img src=".\src\assets\yash-Photoroom.png" alt="" className='w-48 rounded-full' />
                        <p className='text-lg font-semibold '>Yashraj Rajput</p>
                    </div>

                    <div id='card' className='max-w-md  p-4 flex flex-col gap-2 items-center justify-center'>
                        <img src=".\src\assets\vinit.png" alt="" className='w-48' />
                        <p className='text-lg font-semibold '>Vinit Patil</p>
                    </div>

                    <div id='card' className='max-w-md  p-4 flex flex-col gap-2 items-center justify-center'>
                        <img src=".\src\assets\sanket_photo.png" alt="" className='w-48 ' />
                        <p className='text-lg font-semibold '>Sanket Suryavanshi</p>
                    </div>

                    <div id='card' className='max-w-md  p-4 flex flex-col gap-2 items-center justify-center'>
                        <img src=".\src\assets\sahil-Photoroom.png" alt="" className='w-48 rounded-full ' />
                        <p className='text-lg font-semibold '>Sahil Sathe</p>
                    </div>
                </div>
            </div>

        </>
    )
}

export default TeamMember