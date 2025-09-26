#pragma once
#include <vector>
#include <cstring>

namespace orfp
{
	/// <summary>
	/// Представляет из себя буфер фиксированного размера из элементов типа Т.
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
		/// Добавляет элемент в конец буфера, вытесняя элемент из начала буфера.
		/// Т.е. более новые элементы вытесняют более старые элементы.
		/// Принимает params -- параметры конструктора элемента.
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
		/// Возвращает указатель на данные в буфере, если он заполнен. Иначе бросает исключение.
		/// Данные идут в порядке от более поздних в более ранним.
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
		/// true если буфер заполнен, иначе false.
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
