import React from 'react'
import { Link, NavLink } from "react-router-dom";
import ImageUpload from './ImageUpload';
function Hero() {
    return (
        <>
            <div id="hero" className="  bg-gradient-to-b from-purple-100 via-orange-100 to-transprent   ">

                <div id="hero-container"
                    className="max-w-4xl mx-auto px-6 pt-6 pb-32 flex flex-col sm:items-center sm:text-center sm:pt-12 ">

                    <h1 className="text-3xl font-semibold leading-loose mt-40  sm:text-6xl">
                        Underwater Image Dehazing Using Deep Learning
                    </h1>
                    <p className="text-xl mt-4 text-gray-800 sm:text-2xl sm:mt-8 sm:leading-normal">
                        Dive deeper into clarity: Harnessing deep learning to unveil the hidden beauty of underwater realms. From marine research to security surveillance, clear vision beneath the waves
                    </p>
                </div>
            </div>
            <ImageUpload/>
        </>
    )
}

export default Hero