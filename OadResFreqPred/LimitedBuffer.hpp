#pragma once
#include <vector>
#include <cstring>

namespace orfp
{
	/// <summary>
	/// Buffer with fixed capacity of elements of type T.
	/// </summary>
	template<class T>
	class LimitedBuffer final {
	public:
		LimitedBuffer(size_t capacity) : m_buffer(capacity), m_firstElementPtr(m_buffer.data()), m_lastElementPtr(m_firstElementPtr + capacity - 1) {}

		LimitedBuffer(const LimitedBuffer&) = delete;
		LimitedBuffer(LimitedBuffer&&) = delete;

		LimitedBuffer& operator=(const LimitedBuffer&) = delete;
		LimitedBuffer& operator=(LimitedBuffer&&) = delete;

		/// <summary>
		/// Adds element to the end of the buffer.
		/// That is, the newer elements replace the elder elements.
		/// Takes 'params' -- parameters for the elements constructor.
		/// </summary>
		template <class... TArgs>
		void add(TArgs&&... params)
		{
			if (isFilled()) 
			{
				// Call dtor of the first element that will be replaced by the newer element
				m_firstElementPtr->~T();
			}
			else
			{
				++m_size;
			}

			std::memmove(m_firstElementPtr, m_firstElementPtr + 1, (m_buffer.capacity() - 1) * sizeof(T));
			new (m_lastElementPtr) T{ std::forward<TArgs>(params)... };
		}

		/// <summary>
		/// Returns the pointer to data if the buffer is filled. Else, an exception is thrown.
		/// Data go in order from elder to newer added elements.
		/// </summary>
		T* data() 
		{ 
			if (!isFilled()) 
			{
				throw std::exception("Unable to get data from buffer that isn't filled yet");
			}

			return m_buffer.data(); 
		}

		/// <summary>
		/// Checks whether the buffer filled or not.
		/// Recommended to be checked before calling LimitedBuffer::data().
		/// </summary>
		bool isFilled() const { return m_size == m_buffer.capacity(); }

		~LimitedBuffer() = default;

	private:
		std::vector<T> m_buffer;
		size_t m_size{0};
		T* m_firstElementPtr{ nullptr };
		T* m_lastElementPtr{ nullptr };
	};
}
