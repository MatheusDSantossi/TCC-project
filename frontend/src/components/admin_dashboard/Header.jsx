import React from 'react'
import 
 {BsFillBellFill, BsFillEnvelopeFill, BsPersonCircle, BsSearch, BsJustify}
 from 'react-icons/bs'

function Header({OpenSidebar}) {
  return (
    <header className='flex items-center justify-between p-4 bg-white shadow-md'>
      <input type="text" name="search" placeholder='Type to search...' className='w-1/3 p-2 border border-gray-300 rounded-md foucs:outline-none focus:ring-2 foucs:ring-secondary' />
        <div className='menu-icon'>
            <BsJustify className='icon' onClick={OpenSidebar}/>
        </div>
        <div className='flex items-center space-x-4'>
            <span className="text-2xl cursor-pointer">bsBell</span>
            <span className="text-2xl cursor-pointer">bsBell</span>
            <span className="text-2xl cursor-pointer">bsBell</span>
        </div>
    </header>
  )
}

export default Header
